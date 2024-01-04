# The purpose of this empty dialect module is to enable successfully loading the "custom" dialect.
# Without this file here (and a corresponding _cext.globals.append_dialect_search_prefix("custom_dialect")),
# PyGlobals::loadDialectModule would search and fail to find the "custom" dialect for each Operation.create("custom.op")
# (amongst other things).
