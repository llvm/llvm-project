; RUN: opt -annotation-remarks -pass-remarks-missed='annotation-remarks' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --input-file=%t.opt.yaml %s
; RUN: opt -passes='annotation-remarks' -pass-remarks-missed='annotation-remarks' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --input-file=%t.opt.yaml %s

; CHECK:      --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '4'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            _remarks1
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '3'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            _remarks2
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: Function:        test2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '2'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            _remarks1
; CHECK-NEXT: ...

define void @test1(float* %a) {
entry:
  %a.addr = alloca float*, align 8, !annotation !0
  store float* null, float** %a.addr, align 8, !annotation !1
  store float* %a, float** %a.addr, align 8, !annotation !0
  ret void, !annotation !0
}

define void @test2(float* %a) {
entry:
  %a.addr = alloca float*, align 8, !annotation !1
  ret void, !annotation !1
}

!0 = !{!"_remarks1", !"_remarks2"}
!1 = !{!"_remarks1"}
