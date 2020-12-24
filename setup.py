import setuptools

with open("README.md", "r") as fr:
    long_description = fr.read()

setuptools.setup(
    name="corenlp_client", # Replace with your own username
    version="1.0.3",
    author="Jason Fang",
    author_email="jasonfang3900@gmail.com",
    description="A python wapper for Stanford CoreNLP, simple and customizable. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jason3900/corenlp_client",
    packages=setuptools.find_packages(),
    install_requires=["nltk", "requests"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
