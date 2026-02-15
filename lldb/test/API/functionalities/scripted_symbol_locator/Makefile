C_SOURCES := main.c
USE_SYSTEM_STDLIB := 1

# Linux/FreeBSD need --build-id for a UUID; Darwin gets one automatically.
ifneq "$(OS)" "Darwin"
LD_EXTRAS := -Wl,--build-id
endif

include Makefile.rules
