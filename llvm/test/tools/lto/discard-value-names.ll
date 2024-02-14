; RUN: llvm-as %s -o %t.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t.dylib %t.o -save-temps  -undefined dynamic_lookup -exported_symbol _bar -lSystem -mllvm -lto-discard-value-names
; RUN: llvm-dis %t.dylib.lto.opt.bc -o - | FileCheck --check-prefix=DISCARD %s

; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t.dylib %t.o -save-temps  -undefined dynamic_lookup -exported_symbol _bar -lSystem -mllvm -lto-discard-value-names=false
; RUN: llvm-dis %t.dylib.lto.opt.bc -o - | FileCheck --check-prefix=KEEP %s

; The test requires asserts, as it depends on the default value for
; -lto-discard-value-names at the moment.
; FIXME: -lto-discard-value-names is ignored at the moment.

; REQUIRES: asserts

; DISCARD: %cmp.i = icmp
; DISCARD: %add = add i32

; KEEP: %cmp.i = icmp
; KEEP : %add = add i32

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

declare void @external()

define internal i32 @foo(i32 %a, i32 %b) {
entry:
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %then, label %else

then:
  call void @external()
  ret i32 10

else:
  ret i32 20
}

define i32 @bar(i32 %a) {
  %res = call i32 @foo(i32 %a, i32 10)
  %add = add i32 %res, %a
  ret i32 %add
}
