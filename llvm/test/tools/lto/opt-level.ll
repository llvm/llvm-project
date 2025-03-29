; RUN: llvm-as %s -o %t.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -macosx_version_min 10.8.0 -dylib -mllvm -O0 -o %t.dylib %t.o -lSystem
; RUN: llvm-nm --no-llvm-bc %t.dylib | FileCheck --check-prefix=CHECK-O0 %s
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -macosx_version_min 10.8.0 -dylib -mllvm -O2 -o %t.dylib %t.o -lSystem
; RUN: llvm-nm --no-llvm-bc %t.dylib | FileCheck --check-prefix=CHECK-O2 %s

target triple = "x86_64-apple-macosx10.8.0"

; CHECK-O0: t _f1
; CHECK-O2-NOT: _f1
define internal void @f1() {
  ret void
}

; CHECK-O0: T _f2
; CHECK-O2: T _f2
define void @f2() {
  call void @f1()
  ret void
}
