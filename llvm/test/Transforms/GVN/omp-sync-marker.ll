; RUN: opt -passes=gvn,dse -S < %s | FileCheck %s

declare void @ext()

; The marker may write to %p, so %p has to be loaded again after it.
define i32 @reload_after_marker(ptr noalias %p) {
; CHECK-LABEL: @reload_after_marker(
; CHECK-NEXT:    %a = load i32, ptr %p, align 4
; CHECK-NEXT:    call void @llvm.omp.sync.marker.p0(ptr %p)
; CHECK-NEXT:    %b = load i32, ptr %p, align 4
; CHECK-NEXT:    %c = add i32 %a, %b
; CHECK-NEXT:    ret i32 %c
  %a = load i32, ptr %p
  call void @llvm.omp.sync.marker.p0(ptr %p)
  %b = load i32, ptr %p
  %c = add i32 %a, %b
  ret i32 %c
}

; The marker may read %p, so the first store isn't dead.
define void @store_before_marker(ptr noalias %p) {
; CHECK-LABEL: @store_before_marker(
; CHECK-NEXT:    store i32 1, ptr %p, align 4
; CHECK-NEXT:    call void @llvm.omp.sync.marker.p0(ptr %p)
; CHECK-NEXT:    store i32 2, ptr %p, align 4
; CHECK-NEXT:    ret void
  store i32 1, ptr %p
  call void @llvm.omp.sync.marker.p0(ptr %p)
  store i32 2, ptr %p
  ret void
}

; The marker doesn't capture %p, so @ext still can't modify it.
define i32 @no_capture(ptr noalias %p) {
; CHECK-LABEL: @no_capture(
; CHECK-NEXT:    call void @llvm.omp.sync.marker.p0(ptr %p)
; CHECK-NEXT:    %a = load i32, ptr %p, align 4
; CHECK-NEXT:    call void @ext()
; CHECK-NEXT:    %c = add i32 %a, %a
; CHECK-NEXT:    ret i32 %c
  call void @llvm.omp.sync.marker.p0(ptr %p)
  %a = load i32, ptr %p
  call void @ext()
  %b = load i32, ptr %p
  %c = add i32 %a, %b
  ret i32 %c
}

; The marker only accesses the memory its argument points to.
define i32 @other_memory(ptr noalias %p, ptr noalias %q) {
; CHECK-LABEL: @other_memory(
; CHECK-NEXT:    %a = load i32, ptr %q, align 4
; CHECK-NEXT:    call void @llvm.omp.sync.marker.p0(ptr %p)
; CHECK-NEXT:    %c = add i32 %a, %a
; CHECK-NEXT:    ret i32 %c
  %a = load i32, ptr %q
  call void @llvm.omp.sync.marker.p0(ptr %p)
  %b = load i32, ptr %q
  %c = add i32 %a, %b
  ret i32 %c
}
