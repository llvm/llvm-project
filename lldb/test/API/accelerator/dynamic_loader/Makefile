C_SOURCES := main.c

include Makefile.rules

# Build a real shared library the accelerator dynamic loader can load. a.out is
# deliberately not linked against it: the test only needs the .so to exist on
# disk so the loader can pull it into the accelerator target.
all: libgpu_lib.so

libgpu_lib.so: gpu_lib.c
	$(CC) $(CFLAGS) -fPIC -shared -o "$@" $<
