# FIXME: mockSoftTrapRuntime.c shouldn't really be built with -fbounds-safety
C_SOURCES := main.c mockSoftTrapRuntime.c

soft-trap-test-minimal: CFLAGS_EXTRAS := -fbounds-safety -fbounds-safety-soft-traps=call-minimal
soft-trap-test-minimal: all

soft-trap-test-with-str: CFLAGS_EXTRAS := -fbounds-safety -fbounds-safety-soft-traps=call-with-str
soft-trap-test-with-str: all

include Makefile.rules
