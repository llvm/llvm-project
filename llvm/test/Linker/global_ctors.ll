; RUN: llvm-link -S %s %S/Inputs/old_global_ctors.3.4.bc | FileCheck %s
; RUN: llvm-link -S %S/Inputs/old_global_ctors.3.4.bc %s | FileCheck %s

; old_global_ctors.3.4.bc contains the following LLVM IL, assembled into
; bitcode by llvm-as from 3.4.  It uses a two element @llvm.global_ctors array.
; ---
; declare void @a_global_ctor()
; declare void @b_global_ctor()
;
; @llvm.global_ctors = appending global [2 x { i32, void ()* } ] [
;   { i32, void ()* } { i32 65535, void ()* @a_global_ctor },
;   { i32, void ()* } { i32 65535, void ()* @b_global_ctor }
; ]
; ---

declare void @c_global_ctor()
declare void @d_global_ctor()

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* } ] [
  { i32, void ()*, i8* } { i32 65535, void ()* @c_global_ctor, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* @d_global_ctor, i8* null }
]

; CHECK: @llvm.global_ctors = appending global [4 x { i32, void ()*, i8* }] [
; CHECK-DAG:  { i32, void ()*, i8* } { i32 65535, void ()* @a_global_ctor, i8* null }
; CHECK-DAG:  { i32, void ()*, i8* } { i32 65535, void ()* @b_global_ctor, i8* null }
; CHECK-DAG:  { i32, void ()*, i8* } { i32 65535, void ()* @c_global_ctor, i8* null }
; CHECK-DAG:  { i32, void ()*, i8* } { i32 65535, void ()* @d_global_ctor, i8* null }
; CHECK: ]
