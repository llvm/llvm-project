C_SOURCES := main.c
USE_SYSTEM_STDLIB := 1

ifneq "$(OS)" "Darwin"
LD_EXTRAS := -Wl,--build-id
endif

include Makefile.rules
