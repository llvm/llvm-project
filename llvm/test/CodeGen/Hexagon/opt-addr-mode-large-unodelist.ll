; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

; Verify that processAddUses correctly folds an A2_addi into multiple
; store offsets. With many uses, isSafeToExtLR must be called once
; (not once per use) to avoid O(N^2) compile time.

; The A2_addi computing the base address should be eliminated; all
; stores should use the original base register with the combined offset.

; CHECK-LABEL: f0:
; CHECK-NOT: = add(r
; CHECK-DAG: memb(r0+#13)
; CHECK-DAG: memb(r0+#14)
; CHECK-DAG: memb(r0+#15)
; CHECK-DAG: memb(r0+#16)
; CHECK-DAG: memb(r0+#17)
; CHECK-DAG: memb(r0+#18)
; CHECK-DAG: memb(r0+#19)
; CHECK-DAG: memb(r0+#20)

define void @f0(ptr %base, i8 %a, i8 %b, i8 %c, i8 %d,
                i8 %e, i8 %f, i8 %g, i8 %h) nounwind {
entry:
  %p0 = getelementptr inbounds i8, ptr %base, i32 13
  store i8 %a, ptr %p0, align 1
  %p1 = getelementptr inbounds i8, ptr %p0, i32 1
  store i8 %b, ptr %p1, align 1
  %p2 = getelementptr inbounds i8, ptr %p0, i32 2
  store i8 %c, ptr %p2, align 1
  %p3 = getelementptr inbounds i8, ptr %p0, i32 3
  store i8 %d, ptr %p3, align 1
  %p4 = getelementptr inbounds i8, ptr %p0, i32 4
  store i8 %e, ptr %p4, align 1
  %p5 = getelementptr inbounds i8, ptr %p0, i32 5
  store i8 %f, ptr %p5, align 1
  %p6 = getelementptr inbounds i8, ptr %p0, i32 6
  store i8 %g, ptr %p6, align 1
  %p7 = getelementptr inbounds i8, ptr %p0, i32 7
  store i8 %h, ptr %p7, align 1
  ret void
}
