; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=simplify-unconditional-branch --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t

; CHECK_INTERESTINGNESS-LABEL: define void @test_phi1
; CHECK-INTERESTINGNESS: %A = alloca i32
; CHECK-INTERESTINGNESS: store i32 %{{P|V}}, ptr %A

; RESULT-LABEL: define void @test_phi1
; RESULT: entry:
; RESULT-NEXT: %A = alloca i32
; RESULT-NEXT: store i32 %V, ptr %A
; RESULT-NEXT: ret void

define void @test_phi1(i32 %V) {
entry:
  %A = alloca i32
  br label %loop.body
loop.body:
  %P = phi i32 [ %V, %entry ], [ %P, %loop.body ]
  store i32 %P, ptr %A
  br label %loop.body
}

; CHECK_INTERESTINGNESS-LABEL: define void @test_phi2
; CHECK-INTERESTINGNESS: %A = alloca i32
; CHECK-INTERESTINGNESS: store i32 %{{P|V}}, ptr %A

; RESULT-LABEL: define void @test_phi2
; RESULT: entry:
; RESULT-NEXT: %A = alloca i32
; RESULT-NEXT: br i1 %C, label %loop.body, label %load
; RESULT: load:
; RESULT-NEXT: %L = load i32, ptr %A
; RESULT-NEXT: ret void
; RESULT: loop.body:
; RESULT-NEXT: %P = phi i32 [ %V, %entry ]
; RESULT-NEXT: store i32 %P, ptr %A
; RESULT-NEXT: ret void

define void @test_phi2(i1 %C, i32 %V) {
entry:
  %A = alloca i32
  br i1 %C, label %loop.body, label %load
load:
  %L = load i32, ptr %A
  br label %loop.body
loop.body:
  %P = phi i32 [ %V, %entry ], [ %P, %loop.body ], [ %L, %load ]
  store i32 %P, ptr %A
  br label %loop.body
}

; CHECK_INTERESTINGNESS-LABEL: define void @test_phi3
; CHECK-INTERESTINGNESS: %A = alloca i32
; CHECK-INTERESTINGNESS: store i32 %{{P|V}}, ptr %A

; RESULT-LABEL: define void @test_phi3
; RESULT: entry:
; RESULT-NEXT: %A = alloca i32
; RESULT-NEXT: switch i32 %S, label %loop.body [
; RESULT-NEXT:   i32 1, label %loop.body
; RESULT-NEXT:   i32 2, label %loop.body
; RESULT-NEXT:   i32 3, label %load
; RESULT-NEXT: ]
; RESULT: load:
; RESULT-NEXT: %L = load i32, ptr %A
; RESULT-NEXT: ret void
; RESULT: loop.body:
; RESULT-NEXT: %P = phi i32 [ %V, %entry ], [ %V, %entry ], [ %V, %entry ]
; RESULT-NEXT: store i32 %P, ptr %A
; RESULT-NEXT: ret void


define void @test_phi3(i32 %S, i32 %V) {
entry:
  %A = alloca i32
  switch i32 %S, label %loop.body [
    i32 1, label %loop.body
    i32 2, label %loop.body
    i32 3, label %load
  ]
load:
  %L = load i32, ptr %A
  br label %loop.body
loop.body:
  %P = phi i32 [ %V, %entry ], [ %P, %loop.body ], [ %V, %entry ], [ %V, %entry ], [ %L, %load ]
  store i32 %P, ptr %A
  br label %loop.body
}
