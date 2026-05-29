; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=simplify-unconditional-branch --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t

; CHECK_INTERESTINGNESS-LABEL: define void @test_void
; CHECK-INTERESTINGNESS: %A = alloca i32
; CHECK-INTERESTINGNESS: store i32 %V, ptr %A

; RESULT-LABEL: define void @test_void
; RESULT: entry:
; RESULT-NEXT: %A = alloca i32
; RESULT-NEXT: store i32 %V, ptr %A
; RESULT-NEXT: ret void

define void @test_void(i32 %V) {
entry:
  %A = alloca i32
  br label %loop.body
loop.body:
  store i32 %V, ptr %A
  br label %loop.body
}

; CHECK_INTERESTINGNESS-LABEL: define float @test_float
; CHECK-INTERESTINGNESS: %A = alloca float
; CHECK-INTERESTINGNESS: store float %V, ptr %A

; RESULT-LABEL: define float @test_float
; RESULT: entry:
; RESULT-NEXT: %A = alloca float
; RESULT-NEXT: store float %V, ptr %A
; RESULT-NEXT: ret float 0

define float @test_float(float %V) {
entry:
  %A = alloca float
  br label %loop.body
loop.body:
  store float %V, ptr %A
  br label %loop.body
}
