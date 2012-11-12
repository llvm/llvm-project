all: foo.dylib

CWD := $(shell pwd)

all: foo.dylib

foo.dylib:
	clang++ -O0 -g -stdlib=libc++ -dynamiclib -o plugin.dylib plugin.cpp -framework LLDB -F $(LLDB_FRAMEWORK)/..

clean:
	rm -rf plugin.dylib plugin.dylib.dSYM/* plugin.dylib.dSYM
