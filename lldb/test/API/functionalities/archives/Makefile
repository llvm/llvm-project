C_SOURCES := main.c a.c b.c c.c
EXE :=  # Define a.out explicitly
MAKE_DSYM := NO

all: a.out libbar.a libfoo-thin.a

a.out: main.o libfoo.a
	$(LD) $(LDFLAGS) $^ -o $@

libfoo.a: a.o b.o
	$(AR) $(ARFLAGS) $@ $^

# This tests whether lldb can load a thin archive
libbar.a: c.o
	$(eval LLVM_AR := $(LLVM_TOOLS_DIR)/llvm-ar)
	$(eval LLVM_ARFLAGS := -rcsDT)
	$(LLVM_AR) $(LLVM_ARFLAGS) $@ $^

libfoo-thin.a: a.o b.o
	$(eval LLVM_AR := $(LLVM_TOOLS_DIR)/llvm-ar)
	$(eval LLVM_ARFLAGS := -rcsUT)
	$(LLVM_AR) $(LLVM_ARFLAGS) $@ $^

include Makefile.rules
