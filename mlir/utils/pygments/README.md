## Pygments Lexer for MLIR

This file contains a simple Pygments lexer configuration for MLIR, derived from
the version used in the original CGO paper. Pygments allows for advanced
configurable syntax highlighting of any code. This lexer is known to be
incomplete and support mostly core IR with a subset of built-in types.
Additions and customizations are welcome.

### Standalone Usage

Install Pygments, e.g., by running `pip install Pygments` or a Python package
manager of your choosing. Use the standalone `pygmentize` command by
instructing it to load the custom lexer:

```
pygmentize -l /path/to/mlir_lexer.py:MlirLexer -x myfile.mlir
```

This will produce highlighted output in the terminal. Other output formats are
available, see Pygments [documentation](https://pygments.org/docs/) for more
information.

### MkDocs / Python-Markdown

Create a Markdown extension that registers the lexer via Pygments' LEXERS
mapping:

```python
# e.g., docs/pygments/mlir.py
from markdown import Extension
import pygments.lexers._mapping as _mapping

def _register_mlir_lexer():
    if "MlirLexer" not in _mapping.LEXERS:
        _mapping.LEXERS["MlirLexer"] = (
            "your.module.path.mlir_lexer",  # adjust to your project
            "MLIR",
            ("mlir",),
            ("*.mlir",),
            ("text/x-mlir",),
        )

class MlirHighlightExtension(Extension):
    def extendMarkdown(self, md):
        _register_mlir_lexer()

def makeExtension(**kwargs):
    return MlirHighlightExtension(**kwargs)
```

Add to `mkdocs.yml` (before `pymdownx.highlight`):

```yaml
markdown_extensions:
  - your.module.path.mlir
  - pymdownx.highlight
```

### LaTeX Usage

First, make sure your distribution includes the `minted` package and list in
the preamble.

```latex
\usepackage{minted}
```

Place the `mlir_lexer.py` in a place where the `latex` binary can find it,
typically in the working directory next to the main `.tex` file. Note that you
will have to invoke `latex` with the `-shell-escape` flag. See the `minted`
package [documentation](https://ctan.org/pkg/minted?lang=en) for more
information.

Leverage the custom lexer facility of `minted` to use this lexer in your
document as:

```latex
\begin{minted}{mlir_lexer.py:MlirLexer -x}
   ... your code here ...
\end{minted}
```
