ENABLE_THREADS := YES
CXX_SOURCES := main.cpp

# Sources shared between the multithreaded tests live in the parent directory.
override VPATH += $(SRCDIR)/..
override CFLAGS_EXTRAS += -I$(SRCDIR)/..

include Makefile.rules
