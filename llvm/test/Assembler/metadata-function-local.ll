; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | llvm-as --opaque-pointers=0 | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s

declare void @llvm.metadata(metadata)

define void @foo(i32 %arg) {
entry:
  %before = alloca i32
  call void @llvm.metadata(metadata i32 %arg)
  call void @llvm.metadata(metadata i32* %after)
  call void @llvm.metadata(metadata i32* %before)
  %after = alloca i32
  ret void

; CHECK: %before = alloca i32
; CHECK: call void @llvm.metadata(metadata i32 %arg)
; CHECK: call void @llvm.metadata(metadata i32* %after)
; CHECK: call void @llvm.metadata(metadata i32* %before)
; CHECK: %after = alloca i32
}
