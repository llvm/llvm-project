; RUN: opt -passes=ripple -S < %s | FileCheck %s --implicit-check-not="warning:"

; ModuleID = '<stdin>'
source_filename = "<stdin>"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"

; CHECK-LABEL: i32 @f()
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i32 2

define dso_local i32 @f() local_unnamed_addr {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i32(i32 0, i32 32, i32 4, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1)
  %0 = tail call i32 @llvm.ripple.block.index.i32(ptr %BS, i32 0)
  %1 = tail call i32 @llvm.ripple.slice.i32(i32 %0, i64 2, i64 0, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret i32 %1
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i32 @llvm.ripple.block.index.i32(ptr, i32 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i32(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ripple.slice.i32(i32, i64 immarg, i64 immarg, i64 immarg, i64 immarg, i64 immarg, i64 immarg, i64 immarg, i64 immarg, i64 immarg, i64 immarg) #2

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
