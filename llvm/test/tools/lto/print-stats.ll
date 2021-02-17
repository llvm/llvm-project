; RUN: llvm-as %s -o %t.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -dylib -mllvm -stats -o %t.dylib %t.o 2>&1 | FileCheck --check-prefix=STATS %s
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -dylib -o %t.dylib %t.o 2>&1 | FileCheck --check-prefix=NO_STATS %s
; REQUIRES: asserts

target triple = "x86_64-apple-macosx10.8.0"

define i32 @test(i32 %a) {
  %r = add i32 %a, 1
  %r.1 = add i32 1, %a
  %r.2 = add i32 %r, %r.1
  ret i32 %r.2
}

; STATS: Statistics Collected
; NO_STATS-NOT: Statistics Collected
