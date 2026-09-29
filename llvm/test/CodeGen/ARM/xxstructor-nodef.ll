; RUN: llc -mtriple=arm-unknown-linux-gnueabihf < %s | FileCheck %s

; This test contains a llvm.global_ctors with no other definitions. Make sure we do not crash in that case.
; CHECK: .section        .init_array,"aw",%init_array

declare  ccc void @ghczmbignum_GHCziNumziBackendziSelected_init__prof_init()
@llvm.global_ctors = appending global [1 x {i32, void ()*, i8* }] [{i32, void ()*, i8* }{i32  65535, void ()*  @ghczmbignum_GHCziNumziBackendziSelected_init__prof_init, i8*  null } ]
