from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    build_dict={"WINDOWS_DWARF_SYMTAB": "1"},
)
