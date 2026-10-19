#!/usr/bin/env python3

"""Script to link #GH123 to our github issue tracker.

We use MyST and reST, so this Sphinx plugin operates on the shared doctree
representation of the documentation, which is effectively a documentation-AST.

Docutils doctree node reference:
https://docutils.sourceforge.io/docs/ref/doctree.html
"""

import argparse
import re
import sys
import tempfile
import textwrap
from typing import Dict, List, Type, Tuple
from pathlib import Path
from llvm_sphinx.help import venv_help

try:
    from docutils import nodes
    from sphinx.application import Sphinx
except ImportError as err:
    print(venv_help(err), file=sys.stderr)
    raise

__version__ = "1.0"

GH_LINK_RE = re.compile("#GH([1-9][0-9]+)")
GH_LINK_TMPL = "https://github.com/llvm/llvm-project/issues/{}"
SKIP_NODES: Tuple[Type[nodes.Node], ...] = (
    nodes.FixedTextElement,
    nodes.literal,
    nodes.raw,
    nodes.reference,
)


def make_gh_link(issue: str) -> nodes.reference:
    """Create the docutils node that writers render as an external link."""
    return nodes.reference("", "#" + issue, refuri=GH_LINK_TMPL.format(issue))


def replace_gh_links(node: nodes.Text) -> None:
    """Replace one text node with text fragments and GitHub issue links.

    docutils text nodes cannot contain child nodes, so a single string like
    "See #GH123." has to become a sibling list: Text("See "), reference(...),
    Text("."). The parent node owns that sibling list.
    """
    remaining = str(node)
    replacements = []

    while GH_LINK_RE.search(remaining):
        before, issue, remaining = GH_LINK_RE.split(remaining, maxsplit=1)
        if before:
            replacements.append(nodes.Text(before))
        replacements.append(make_gh_link(issue))

    # If we found matches, do the replacement.
    if replacements:
        if remaining:
            replacements.append(nodes.Text(remaining))
        node.parent.replace(node, replacements)


def replace_gh_links_in_subtree(node: nodes.Node) -> None:
    """Rewrite linkable text nodes under node, pruning ignored subtrees.

    The doctree is the markup-language neutral documentation AST, so it
    handles both reST and MyST (markdown). It helps us avoid rewriting #GH123
    in code and literal blocks.
    """
    # DFS cutoff for code blocks etc.
    if isinstance(node, SKIP_NODES):
        return

    # Find #GH123 links in text blocks and linkify them.
    if isinstance(node, nodes.Text):
        if GH_LINK_RE.search(str(node)):
            replace_gh_links(node)
        return

    # Recursive DFS traversal of children.
    if isinstance(node, nodes.Element):
        for child in list(node.children):
            replace_gh_links_in_subtree(child)


def subst_gh_links(_app: Sphinx, doctree: nodes.document) -> None:
    """Link #GH123 references after the source has been parsed."""
    replace_gh_links_in_subtree(doctree)


def setup(app: Sphinx) -> Dict[str, object]:
    app.connect("doctree-read", subst_gh_links)
    return dict(version=__version__, parallel_read_safe=True, parallel_write_safe=True)


# -----------------------------------------------------------------------------
# Test code only below:
# -----------------------------------------------------------------------------
def _build_test_docs() -> Tuple[str, str]:
    """Build in-file reST and Markdown test strings and return their HTML."""
    srcdir = Path(__file__).resolve().parent / "test"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        outdir = tmp_path / "out"
        doctreedir = tmp_path / "doctrees"
        app = Sphinx(
            srcdir=srcdir,
            confdir=srcdir,
            outdir=outdir,
            doctreedir=doctreedir,
            buildername="html",
            freshenv=True,
            warningiserror=True,
            status=None,
            warning=None,
        )
        app.build()
        return (
            (outdir / "rest.html").read_text(encoding="utf-8"),
            (outdir / "markdown.html").read_text(encoding="utf-8"),
        )


def _issue_href(issue: str) -> str:
    return f'href="{GH_LINK_TMPL.format(issue)}"'


def _check_contains(html: str, needle: str) -> None:
    if needle not in html:
        raise AssertionError(f"expected HTML to contain: {needle}")


def run_tests() -> None:
    rest_html, markdown_html = _build_test_docs()

    for issue in ("123", "456"):
        _check_contains(rest_html, _issue_href(issue))
    for issue in ("0123", "333", "666", "777", "888"):
        _check_contains(rest_html, f"#GH{issue}")
    _check_contains(rest_html, 'href="https://example.com/rst"')

    for issue in ("234", "567"):
        _check_contains(markdown_html, _issue_href(issue))
    for issue in ("0234", "444", "999", "778"):
        _check_contains(markdown_html, f"#GH{issue}")
    _check_contains(markdown_html, 'href="https://example.com/md"')
