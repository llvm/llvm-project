; RUN: llvm-extract -S --bb "_Z6kernelv.extracted:%5" < %s | FileCheck %s

; CHECK: define dso_local void @_Z6kernelv.extracted.extracted(i64 %0, i64 %1) {

; CHECK       2:
; CHECK:        %3 = add nuw nsw i64 %0, 1
; CHECK-NEXT:   %4 = sub nuw nsw i64 %3, %1
; CHECK-NEXT:   br label %.exitStub

define dso_local void @_Z6kernelv.extracted(i64 %0, ptr %.out) #0 {
newFuncRoot:
  br label %1

1:
  %2 = phi i64 [ 0, %newFuncRoot ], [ %3, %1 ]
  %3 = add nuw nsw i64 %2, 1
  %4 = icmp eq i64 %2, %3
  br i1 %4, label %5, label %1

5:
  %6 = add nuw nsw i64 %0, 1
  %7 = sub nuw nsw i64 %6, %3
  br label %8

8:
  %9 = add nuw i64 %0, 2
  ret void
}
