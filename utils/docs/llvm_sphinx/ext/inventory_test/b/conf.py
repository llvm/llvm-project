# Test project "b": links to the other project with an `inv:` link.
project = "b"
extensions = ["myst_parser", "sphinx.ext.intersphinx", "llvm_sphinx.ext.inventory"]
intersphinx_mapping = {
    "a": ("https://a.test/", None),
    "b": ("https://b.test/", None),
}
