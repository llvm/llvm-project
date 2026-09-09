extensions = ["llvm_sphinx.ext.absolute_links", "myst_parser"]
master_doc = "index"
project = "absolute links test"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
llvm_sphinx_doc_url_prefixes = (
    "https://example.test/docs/",
    "https://www.example.test/docs/",
)
