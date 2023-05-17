; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; Test TransformForPostIncUse and LSR's expansion of expressions in
; post-inc form to ensure the implementation can handle expressions
; DAGs, not just trees.

target triple = "x86_64-apple-darwin"

; Verify that -loop-reduce runs without "hanging" and reuses post-inc
; expansions.
; CHECK-LABEL: @test(
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK: icmp
; CHECK-NOT: icmp
define void @test(ptr %base, i32 %a0) nounwind {
entry:
  br label %bb1
bb1:
  %n0 = sub i32 0, %a0
  %t0 = icmp ugt i32 %n0, -4
  %m0 = select i1 %t0, i32 %n0, i32 -4
  %a1 = add i32 %m0, %a0
  %n1 = sub i32 0, %a1
  %t1 = icmp ugt i32 %n1, -4
  %m1 = select i1 %t1, i32 %n1, i32 -4
  %a2 = add i32 %m1, %a1
  %n2 = sub i32 0, %a2
  %t2 = icmp ugt i32 %n2, -4
  %m2 = select i1 %t2, i32 %n2, i32 -4
  %a3 = add i32 %m2, %a2
  %n3 = sub i32 0, %a3
  %t3 = icmp ugt i32 %n3, -4
  %m3 = select i1 %t3, i32 %n3, i32 -4
  %a4 = add i32 %m3, %a3
  %n4 = sub i32 0, %a4
  %t4 = icmp ugt i32 %n4, -4
  %m4 = select i1 %t4, i32 %n4, i32 -4
  %a5 = add i32 %m4, %a4
  %n5 = sub i32 0, %a5
  %t5 = icmp ugt i32 %n5, -4
  %m5 = select i1 %t5, i32 %n5, i32 -4
  %a6 = add i32 %m5, %a5
  %n6 = sub i32 0, %a6
  %t6 = icmp ugt i32 %n6, -4
  %m6 = select i1 %t6, i32 %n6, i32 -4
  %a7 = add i32 %m6, %a6
  %n7 = sub i32 0, %a7
  %t7 = icmp ugt i32 %n7, -4
  %m7 = select i1 %t7, i32 %n7, i32 -4
  %a8 = add i32 %m7, %a7
  %n8 = sub i32 0, %a8
  %t8 = icmp ugt i32 %n8, -4
  %m8 = select i1 %t8, i32 %n8, i32 -4
  %a9 = add i32 %m8, %a8
  %n9 = sub i32 0, %a9
  %t9 = icmp ugt i32 %n9, -4
  %m9 = select i1 %t9, i32 %n9, i32 -4
  %a10 = add i32 %m9, %a9
  %n10 = sub i32 0, %a10
  %t10 = icmp ugt i32 %n10, -4
  %m10 = select i1 %t10, i32 %n10, i32 -4
  %a11 = add i32 %m10, %a10
  %n11 = sub i32 0, %a11
  %t11 = icmp ugt i32 %n11, -4
  %m11 = select i1 %t11, i32 %n11, i32 -4
  %a12 = add i32 %m11, %a11
  %n12 = sub i32 0, %a12
  %t12 = icmp ugt i32 %n12, -4
  %m12 = select i1 %t12, i32 %n12, i32 -4
  %a13 = add i32 %m12, %a12
  %n13 = sub i32 0, %a13
  %t13 = icmp ugt i32 %n13, -4
  %m13 = select i1 %t13, i32 %n13, i32 -4
  %a14 = add i32 %m13, %a13
  %n14 = sub i32 0, %a14
  %t14 = icmp ugt i32 %n14, -4
  %m14 = select i1 %t14, i32 %n14, i32 -4
  %a15 = add i32 %m14, %a14
  %n15 = sub i32 0, %a15
  %t15 = icmp ugt i32 %n15, -4
  %m15 = select i1 %t15, i32 %n15, i32 -4
  %a16 = add i32 %m15, %a15
  %gep = getelementptr i8, ptr %base, i32 %a16
  %ofs = add i32 %a16, 4
  %limit = getelementptr i8, ptr %base, i32 %ofs
  br label %loop

loop:
  %iv = phi ptr [ %gep, %bb1 ], [ %inc, %loop ]
  %inc = getelementptr inbounds i8, ptr %iv, i64 1
  %exitcond = icmp eq ptr %inc, %limit
  br i1 %exitcond, label %loop, label %exit

exit:
  ret void
}
