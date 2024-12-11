; RUN: opt -passes='annotation-remarks' -pass-remarks-missed='annotation-remarks' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --input-file=%t.opt.yaml %s
; REQUIRES: apple-disclosure-ios

; CHECK: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: Function:        ptr_bound_compare
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '3'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            bounds-safety-check-ptr-lt-upper-bound
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: Function:        ptr_bound_compare
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '3'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: Function:        ptr_bound_compare
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '5'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            bounds-safety-total-summary
; CHECK-NEXT: ...

define void @ptr_bound_compare([2 x i64] %arg, i32 %arg1) {
bb:
  %extractvalue = extractvalue [2 x i64] %arg, 0
  %inttoptr = inttoptr i64 %extractvalue to ptr
  %extractvalue2 = extractvalue [2 x i64] %arg, 1
  %inttoptr3 = inttoptr i64 %extractvalue2 to ptr
  %icmp = icmp eq i32 %arg1, 0
  br i1 %icmp, label %bb5, label %bb4

bb4:                                              ; preds = %bb
  %zext = zext i32 %arg1 to i64
  br label %bb6

bb5:                                              ; preds = %bb10, %bb
  ret void

bb6:                                              ; preds = %bb10, %bb4
  %phi = phi i64 [ 0, %bb4 ], [ %add, %bb10 ]
  %getelementptr = getelementptr i32, ptr %inttoptr, i64 %phi
  %icmp7 = icmp ult ptr %getelementptr, %inttoptr3, !annotation !0
  %icmp8 = icmp uge ptr %getelementptr, %inttoptr, !annotation !1
  %and = and i1 %icmp7, %icmp8, !annotation !1
  br i1 %and, label %bb10, label %bb9, !annotation !0

bb9:                                              ; preds = %bb6
  tail call void @llvm.ubsantrap(i8 25), !annotation !2
  unreachable

bb10:                                             ; preds = %bb6
  store i32 1, ptr %getelementptr, align 4
  %add = add nuw nsw i64 %phi, 1
  %icmp11 = icmp eq i64 %add, %zext
  br i1 %icmp11, label %bb5, label %bb6
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.ubsantrap(i8 immarg) #0

attributes #0 = { cold noreturn nounwind }

!0 = !{!"bounds-safety-check-ptr-lt-upper-bound"}
!1 = !{!"bounds-safety-check-ptr-ge-lower-bound"}
!2 = !{!"bounds-safety-check-ptr-lt-upper-bound", !"bounds-safety-check-ptr-ge-lower-bound"}
