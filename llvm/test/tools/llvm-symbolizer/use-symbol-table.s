# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64 -g %s -o %t.o

## --use-symbol-table=true is used by old asan_symbolize.py and Android ndk
## ndk-stack.py. Keep it as a no-op compatibility option for a while.
# RUN: llvm-symbolizer --use-symbol-table=true %t.o
