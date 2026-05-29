"""Splice a `select-function<fn=...>` pass at the head of every `-passes=`
pipeline so only the named functions (and their transitive dependencies) are
compiled by `opt`. Driven by `--param fn=NAMES`."""

from lit.llvm.fn_param import add_capture_sub, parse_fn_names


def install(config, lit_config):
    names = parse_fn_names(lit_config)
    if not names:
        return
    sel = "select-function<" + ";".join("fn=" + n for n in names) + ">"
    # -passes='...' / -passes="..." — splice select-function after the quote
    add_capture_sub(config, r"""-passes=(['"])""", r"-passes=\1" + sel + ",")
    # -passes=word (unquoted) — wrap to protect angle brackets
    add_capture_sub(config, r"-passes=([^'\"\s]\S*)", r"-passes='" + sel + r",\1'")
