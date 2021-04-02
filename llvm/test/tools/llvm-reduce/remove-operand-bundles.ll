; Test that llvm-reduce can remove uninteresting operand bundles from calls.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-ALL: declare void @f1()
; CHECK-ALL: declare void @f2()
; CHECK-ALL: declare void @f3()
declare void @f1()
declare void @f2()
declare void @f3()

; CHECK-FINAL-LABEL: define void @interesting(i32 %arg0, i32 %arg1, i32 %arg2) {
; CHECK-FINAL-NEXT:  entry:
; CHECK-FINAL-NEXT:    call void @f1() [ "bundle0"(), "align"(i32 %arg0), "whatever0"() ]
; CHECK-FINAL-NEXT:    call void @f2()
; CHECK-FINAL-NEXT:    call void @f3() [ "align"(i32 %arg2) ]
; CHECK-FINAL-NEXT:    ret void
; CHECK-FINAL-NEXT:  }
define void @interesting(i32 %arg0, i32 %arg1, i32 %arg2) {
entry:
; CHECK-INTERESTINGNESS-LABEL: @interesting(

; CHECK-INTERESTINGNESS: call void @f1()
; CHECK-INTERESTINGNESS: "bundle0"()
; CHECK-INTERESTINGNESS: "align"(i32 %arg0)
; CHECK-INTERESTINGNESS: "whatever0"()

; CHECK-INTERESTINGNESS: call void @f2()

; CHECK-INTERESTINGNESS: call void @f3()
; CHECK-INTERESTINGNESS: "align"(i32 %arg2)

; CHECK-INTERESTINGNESS: ret

  call void @f1() [ "bundle0"(),        "align"(i32 %arg0), "whatever0"() ]
  call void @f2() [ "align"(i32 %arg1), "whatever1"(),      "bundle1"() ]
  call void @f3() [ "whatever2"(),      "bundle2"(),        "align"(i32 %arg2) ]
  ret void
}
