"""Splice a `select-function<fn=...>` pass at the head of every `-passes=`
pipeline so only the named functions (and their transitive dependencies) are
compiled by `opt`. Driven by `--param fn=NAMES`."""

from lit.TestingConfig import SubstituteCaptures


def install(config, lit_config):
    fn = lit_config.params.get("fn")
    if not fn:
        return
    names = [n.strip() for n in fn.split(",") if n.strip()]
    if not names:
        return
    sel = "select-function<" + ";".join("fn=" + n for n in names) + ">"
    # -passes='...' / -passes="..." — splice select-function after the quote
    config.substitutions.append(
        (r"""-passes=(['"])""", SubstituteCaptures(r"-passes=\1" + sel + ","))
    )
    # -passes=word (unquoted) — wrap to protect angle brackets
    config.substitutions.append(
        (r"-passes=([^'\"\s]\S*)", SubstituteCaptures(r"-passes='" + sel + r",\1'"))
    )
