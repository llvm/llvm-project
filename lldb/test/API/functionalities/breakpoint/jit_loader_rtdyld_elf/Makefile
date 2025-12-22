CXX_SOURCES := jitbp.cpp

include Makefile.rules

jitbp.ll: jitbp.cpp
	$(CXX) -g -S -emit-llvm --target=x86_64-unknown-unknown-elf \
	      -o $@ $<

all: jitbp.ll

clean::
	rm -f jitbp.ll
