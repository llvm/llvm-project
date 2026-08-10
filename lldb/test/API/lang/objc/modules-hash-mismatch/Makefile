OBJC_SOURCES := main.m
CFLAGS_EXTRAS = -I$(BUILDDIR)
USE_PRIVATE_MODULE_CACHE = YES

.PHONY: update-module

all: $(EXE)
	"$(MAKE)" -f $(SRCDIR)/Makefile update-module

include Makefile.rules

update-module:
	echo "forcing an update of f.pcm"
	echo "typedef int something_other;" > $(BUILDDIR)/f.h
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) \
		-c $(SRCDIR)/other.m -o $(BUILDDIR)/other.o
