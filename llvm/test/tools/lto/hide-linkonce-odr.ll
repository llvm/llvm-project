; RUN: llvm-as %s -o %t.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macosx_version_min 10.10.0 -o %t.dylib %t.o -save-temps  -undefined dynamic_lookup -exported_symbol _c -exported_symbol _b  -exported_symbol _GlobLinkonce -lSystem

; RUN: llvm-dis %t.dylib.lto.opt.bc -o - | FileCheck --check-prefix=IR %s
; check that @a is no longer a linkonce_odr definition
; IR-NOT: define linkonce_odr void @a()
; check that @b is appended in llvm.used
; IR: @llvm.compiler.used = appending global [2 x i8*] [i8* bitcast ([1 x i8*]* @GlobLinkonce to i8*), i8* bitcast (void ()* @b to i8*)], section "llvm.metadata"

; RUN: llvm-nm %t.dylib | FileCheck --check-prefix=NM %s
; check that the linker can hide @a but not @b, nor @GlobLinkonce
; NM:  S _GlobLinkonce
; NM:  t _a
; NM:  T _b
; NM:  T _c


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

declare void @external()

@GlobLinkonce = linkonce_odr unnamed_addr constant [1 x i8*] [i8* null], align 8

define linkonce_odr void @a() noinline {
  %use_of_GlobLinkonce = load [1 x i8*], [1 x i8*] *@GlobLinkonce
  call void @external()
  ret void
}

define linkonce_odr void @b() {
  ret void
}

define void()* @c() {
  call void @a()
  ret void()* @b
}
