C_SOURCES := main.c

ifeq "$(OS)" ""
    OS = $(shell uname -s)
endif

ifeq "$(OS)" "Darwin"
    LD_EXTRAS = -Xlinker -dead_strip
else
    CFLAGS_EXTRAS += -fdata-sections -ffunction-sections
    ifeq "$(OS)" "Windows_NT"
        LD_EXTRAS = -Xlinker /OPT:REF
    else
        LD_EXTRAS = -Wl,--gc-sections
    endif
endif

MAKE_DSYM := NO

include Makefile.rules
