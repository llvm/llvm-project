; REQUIRES: aarch64-registered-target
; libLTO.dylib shouldn't assert on invalid inputs.
; RUN: llvm-mc -triple=arm64-apple-ios7.0.0 -filetype=obj -o %t.o
; RUN: llvm-ar r %t.a %t.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -dylib -mllvm -O0 -o %t.dylib %t.a
