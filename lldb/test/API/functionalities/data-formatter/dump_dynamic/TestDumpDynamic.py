from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    makefile_extras="""
ifeq "$(OS)" "Windows_NT"
	ifneq "$(MAKE_PDB)" "YES"
		LD_EXTRAS += -Wl,-debug:symtab
	endif
endif
""",
)
