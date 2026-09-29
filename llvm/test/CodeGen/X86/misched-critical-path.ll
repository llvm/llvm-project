; RUN: llc < %s -mtriple=x86_64-apple-darwin8 -misched-print-dags -o - 2>&1 > /dev/null | FileCheck %s
; REQUIRES: asserts

@sc = common global i8 0
@uc = common global i8 0
@ui = common global i32 0

; Regression Test for PR92368.
;
; CHECK: SU(8):   CMP8rr %4:gr8, %3:gr8, implicit-def $eflags
; CHECK:   Predecessors:
; CHECK-NEXT:    SU(6): Data Latency=0 Reg=%4
; CHECK-NEXT:    SU(7): Out  Latency=0
; CHECK-NEXT:    SU(5): Out  Latency=0
; CHECK-NEXT:    SU(3): Data Latency=4 Reg=%3
define void @misched_bug() nounwind {
entry:
  %v0 = load i8, ptr @sc, align 1
  %v1 = zext i8 %v0 to i32
  %v2 = load i8, ptr @uc, align 1
  %v3 = zext i8 %v2 to i32
  %v4 = trunc i32 %v3 to i8
  %v5 = trunc i32 %v1 to i8
  %pair74 = cmpxchg ptr @sc, i8 %v4, i8 %v5 monotonic monotonic
  %v6 = extractvalue { i8, i1 } %pair74, 0
  %v7 = icmp eq i8 %v6, %v4
  %v8 = zext i1 %v7 to i8
  %v9 = zext i8 %v8 to i32
  store i32 %v9, ptr @ui, align 4
  br label %return

return:                                           ; preds = %ventry
  ret void
}

