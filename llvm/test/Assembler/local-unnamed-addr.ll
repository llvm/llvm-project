; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s

; CHECK: @c = local_unnamed_addr constant i32 0
@c = local_unnamed_addr constant i32 0

; CHECK: @a = local_unnamed_addr alias i32, i32* @c
@a = local_unnamed_addr alias i32, i32* @c

; CHECK: define void @f() local_unnamed_addr {
define void @f() local_unnamed_addr {
  ret void
}
