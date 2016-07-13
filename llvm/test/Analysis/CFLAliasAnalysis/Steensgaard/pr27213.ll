; RUN: opt < %s -disable-basicaa -cfl-steens-aa -aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL:     Function: foo
; CHECK: MayAlias: i32* %A, i32* %B
define void @foo(i32* %A, i32* %B) {
entry:
  store i32 0, i32* %A, align 4
  store i32 0, i32* %B, align 4
  ret void
}

; CHECK-LABEL:     Function: bar
; CHECK: MayAlias: i32* %A, i32* %B
; CHECK: MayAlias: i32* %A, i32* %arrayidx
; CHECK: MayAlias: i32* %B, i32* %arrayidx
define void @bar(i32* %A, i32* %B) {
entry:
  store i32 0, i32* %A, align 4
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 1
  store i32 0, i32* %arrayidx, align 4
  ret void
}

@G = global i32 0

; CHECK-LABEL:     Function: baz
; CHECK: MayAlias: i32* %A, i32* @G
define void @baz(i32* %A) {
entry:
  store i32 0, i32* %A, align 4
  store i32 0, i32* @G, align 4
  ret void
}

; CHECK-LABEL: Alias Analysis Evaluator Report
; CHECK: 5 Total Alias Queries Performed
; CHECK: 0 no alias responses
; CHECK: 5 may alias responses
