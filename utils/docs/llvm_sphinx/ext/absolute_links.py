# -*- coding: utf-8 -*-

"""Check for absolute links to documents in the current Sphinx project."""

import io
import sys
import tempfile
from pathlib import Path
from typing import Collection, Dict, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from llvm_sphinx.help import venv_help

try:
    from docutils import nodes
    from sphinx.application import Sphinx
    from sphinx.util import logging
except ImportError as err:
    print(venv_help(err), file=sys.stderr)
    raise

__version__ = "1.0"

logger = logging.getLogger("llvm_sphinx.ext.absolute_links")


def setup(app: Sphinx) -> Dict[str, object]:
    app.add_config_value("llvm_sphinx_doc_url_prefixes", (), "env", [list, tuple])
    app.connect("doctree-read", check_absolute_doc_links)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


def _url_prefix_parts(prefix: str) -> Optional[Tuple[str, str]]:
    try:
        parsed = urlsplit(prefix)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    if port not in (None, 80, 443):
        return None
    path = parsed.path
    if not path.endswith("/"):
        path += "/"
    return parsed.hostname.lower(), path


def _docname_from_url(
    uri: str, prefixes: Sequence[str], found_docs: Collection[str]
) -> Optional[str]:
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or port not in (None, 80, 443)
    ):
        return None

    hostname = parsed.hostname.lower()
    path = unquote(parsed.path)
    for prefix in prefixes:
        prefix_parts = _url_prefix_parts(prefix)
        if prefix_parts is None:
            continue
        prefix_hostname, prefix_path = prefix_parts
        if hostname != prefix_hostname or not path.startswith(prefix_path):
            continue

        relative_path = path[len(prefix_path) :]
        if not relative_path:
            docname = "index"
        elif relative_path.endswith(".html"):
            docname = relative_path[: -len(".html")]
        elif relative_path.endswith("/"):
            docname = relative_path + "index"
        else:
            continue

        if docname in found_docs:
            return docname
    return None


def check_absolute_doc_links(app: Sphinx, doctree: nodes.document) -> None:
    """Diagnose absolute URLs that name documents in this Sphinx project."""
    prefixes = app.config.llvm_sphinx_doc_url_prefixes
    if not prefixes:
        return

    for node in doctree.findall(nodes.reference):
        uri = node.get("refuri")
        if not uri:
            continue
        docname = _docname_from_url(uri, prefixes, app.env.found_docs)
        if docname is None:
            continue
        logger.warning(
            "absolute URL points to document %r in this Sphinx project; "
            "use an internal 'doc' or 'ref' role instead: %s",
            docname,
            uri,
            location=node,
            type="llvm_sphinx",
            subtype="absolute-doc-link",
        )


# -----------------------------------------------------------------------------
# Test code only below:
# -----------------------------------------------------------------------------
def _build_test_docs() -> str:
    """Build the reST and Markdown test inputs and return Sphinx warnings."""
    srcdir = Path(__file__).resolve().parent / "absolute_links_test"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        warnings = io.StringIO()
        app = Sphinx(
            srcdir=srcdir,
            confdir=srcdir,
            outdir=tmp_path / "out",
            doctreedir=tmp_path / "doctrees",
            buildername="html",
            freshenv=True,
            warningiserror=False,
            status=None,
            warning=warnings,
        )
        app.build()
        return warnings.getvalue()


def run_tests() -> None:
    warnings = _build_test_docs()
    expected_urls = (
        "https://example.test/docs/target.html#target-section",
        "https://example.test/docs/",
        "http://www.example.test/docs/target.html?view=1#target-section",
    )
    for url in expected_urls:
        if url not in warnings:
            raise AssertionError(f"expected a warning for {url}")
    if warnings.count("absolute URL points to document") != len(expected_urls):
        raise AssertionError(f"unexpected Sphinx warnings:\n{warnings}")
