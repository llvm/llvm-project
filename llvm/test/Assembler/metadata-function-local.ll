; RUN: llvm-as  < %s | llvm-dis  | llvm-as  | llvm-dis  | FileCheck %s
; RUN: verify-uselistorder  %s

declare void @llvm.metadata(metadata)

define void @foo(i32 %arg) {
entry:
  %before = alloca i32
  call void @llvm.metadata(metadata i32 %arg)
  call void @llvm.metadata(metadata ptr %after)
  call void @llvm.metadata(metadata ptr %before)
  %after = alloca i32
  ret void

; CHECK: %before = alloca i32
; CHECK: call void @llvm.metadata(metadata i32 %arg)
; CHECK: call void @llvm.metadata(metadata ptr %after)
; CHECK: call void @llvm.metadata(metadata ptr %before)
; CHECK: %after = alloca i32
}
