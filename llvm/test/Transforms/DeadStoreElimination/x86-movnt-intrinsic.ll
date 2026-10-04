; RUN: opt -S -passes=dse,gvn < %s | FileCheck %s

; llvm.x86.movnt has no memory attributes, so it is not optimized as an
; ordinary store: an overwritten call is not removed, and its value is not
; forwarded to a later load.

define void @overwritten(ptr %p, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: @overwritten(
; CHECK-NEXT:    call void @llvm.x86.movnt.v2i64(ptr [[P:%.*]], <2 x i64> [[A:%.*]])
; CHECK-NEXT:    store <2 x i64> [[B:%.*]], ptr [[P]], align 16
; CHECK-NEXT:    ret void
;
  call void @llvm.x86.movnt.v2i64(ptr %p, <2 x i64> %a)
  store <2 x i64> %b, ptr %p, align 16
  ret void
}

define i64 @not_forwarded(ptr %p) {
; CHECK-LABEL: @not_forwarded(
; CHECK-NEXT:    call void @llvm.x86.movnt.i64(ptr [[P:%.*]], i64 5)
; CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[P]], align 8
; CHECK-NEXT:    ret i64 [[V]]
;
  call void @llvm.x86.movnt.i64(ptr %p, i64 5)
  %v = load i64, ptr %p, align 8
  ret i64 %v
}
