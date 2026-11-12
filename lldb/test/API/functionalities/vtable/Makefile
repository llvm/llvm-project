CXX_SOURCES := main.cpp

ifeq "$(OS)" "Darwin"
    # Make vtables writable for test_overwrite_vtable test
    # The -no_data_const flag prevents vtables from being placed in __DATA_CONST
    LD_EXTRAS := -Wl,-no_data_const
endif

include Makefile.rules
