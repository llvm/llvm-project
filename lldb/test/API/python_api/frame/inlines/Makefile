C_SOURCES := inlines.c

ifneq (,$(findstring icc,$(CC)))
    CFLAGS_EXTRAS := -debug inline-debug-info
endif

include Makefile.rules
