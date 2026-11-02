; NOTE: Do not autogenerate. Check stable measured identities and unmeasured
; preheaders without depending on metadata node numbering.
; RUN: opt -passes=loop-simplify,verify -S %s | FileCheck %s
;
; Each independent loop after h0 needs a preheader. All original observations
; survive the batch, and newly created blocks have no measured count.
;
; CHECK-LABEL: define void @_Z14divergent_loopPVi(
; CHECK-SAME: !wave.profile ![[TABLE:[0-9]+]]
; CHECK: h0:
; CHECK-NEXT: br i1 %repeat, label %h0, label %h1.preheader, !wave.profile.block ![[H0:[0-9]+]]
; CHECK: h1.preheader:
; CHECK-NEXT: br label %h1, !wave.profile.block ![[PRE1:[0-9]+]]
; CHECK: h1:
; CHECK-NEXT: br i1 %repeat, label %h1, label %h2.preheader, !wave.profile.block ![[H1:[0-9]+]]
; CHECK: h2.preheader:
; CHECK-NEXT: br label %h2, !wave.profile.block ![[PRE2:[0-9]+]]
; CHECK: h2:
; CHECK-NEXT: br i1 %repeat, label %h2, label %exit, !wave.profile.block ![[H2:[0-9]+]]
; CHECK: exit:
; CHECK-NEXT: ret void, !wave.profile.block ![[EXIT:[0-9]+]]

define void @_Z14divergent_loopPVi(i1 %repeat) !wave.profile !0 {
entry:
  br label %h0, !wave.profile.block !1
h0:
  br i1 %repeat, label %h0, label %h1, !wave.profile.block !2
h1:
  br i1 %repeat, label %h1, label %h2, !wave.profile.block !3
h2:
  br i1 %repeat, label %h2, label %exit, !wave.profile.block !4
exit:
  ret void, !wave.profile.block !5
}

; CHECK: ![[TABLE]] = distinct !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 1, i64 1, i64 0, i64 0}
; CHECK: ![[H0]] = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 [[ID1:[0-9]+]]}
; CHECK: ![[PRE1]] = !{i64 2, i64 2480672276464841217, i64 [[ID1]], i64 0, i64 2}
; CHECK: ![[H1]] = !{i64 2, i64 2480672276464841217, i64 2, i64 1, i64 2, i64 [[ID2:[0-9]+]]}
; CHECK: ![[PRE2]] = !{i64 2, i64 2480672276464841217, i64 [[ID2]], i64 0, i64 3}
; CHECK: ![[H2]] = !{i64 2, i64 2480672276464841217, i64 3, i64 1, i64 3, i64 4}
; CHECK: ![[EXIT]] = !{i64 2, i64 2480672276464841217, i64 4, i64 1}

!0 = distinct !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 1, i64 1}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 1, i64 2, i64 3}
!4 = !{i64 2, i64 2480672276464841217, i64 3, i64 1, i64 3, i64 4}
!5 = !{i64 2, i64 2480672276464841217, i64 4, i64 1}
