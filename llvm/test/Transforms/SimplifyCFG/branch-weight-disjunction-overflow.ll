; RUN: opt -passes=simplifycfg -S %s | FileCheck %s
;
; Minimized regression for repeated branch-weight disjunction in SimplifyCFG.
; We build a test-and-set ladder that SimplifyCFG merges through
; mergeConditionalStores. The resulting synthesized branch must keep !prof.
;
; CHECK-LABEL: define void @test(
; CHECK-LABEL: entry:
; CHECK: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}, !prof ![[BW:[0-9]+]]
; CHECK: ![[BW]] = !{!"branch_weights", i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}}

define void @test(i1 %c0, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5, i1 %c6, i1 %c7, i1 %c8, i1 %c9, ptr %p) {
entry:
  br i1 %c0, label %s0, label %n0, !prof !0

s0:
  store i32 1, ptr %p, align 4
  br label %n0

n0:
  br i1 %c1, label %s1, label %n1, !prof !0

s1:
  store i32 1, ptr %p, align 4
  br label %n1

n1:
  br i1 %c2, label %s2, label %n2, !prof !0

s2:
  store i32 1, ptr %p, align 4
  br label %n2

n2:
  br i1 %c3, label %s3, label %n3, !prof !0

s3:
  store i32 1, ptr %p, align 4
  br label %n3

n3:
  br i1 %c4, label %s4, label %n4, !prof !0

s4:
  store i32 1, ptr %p, align 4
  br label %n4

n4:
  br i1 %c5, label %s5, label %n5, !prof !0

s5:
  store i32 1, ptr %p, align 4
  br label %n5

n5:
  br i1 %c6, label %s6, label %n6, !prof !0

s6:
  store i32 1, ptr %p, align 4
  br label %n6

n6:
  br i1 %c7, label %s7, label %n7, !prof !0

s7:
  store i32 1, ptr %p, align 4
  br label %n7

n7:
  br i1 %c8, label %s8, label %n8, !prof !0

s8:
  store i32 1, ptr %p, align 4
  br label %n8

n8:
  br i1 %c9, label %s9, label %exit, !prof !0

s9:
  store i32 1, ptr %p, align 4
  br label %exit

exit:
  ret void
}

!0 = !{!"branch_weights", i32 2000, i32 0}
