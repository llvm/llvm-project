; RUN: llc < %s -mtriple=s390x-linux-gnu -O3 -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -O3 -mcpu=z14 | FileCheck %s
;
; Test folding of spilled immediate loads and compares.

define i32 @fun0(ptr %src, i32 %arg) nounwind {
; CHECK-LABEL: fun0:
; CHECK: 	mvhi	160(%r15), 0            # 4-byte Folded Spill
; CHECK:	mvc	160(4,%r15), 0(%r2)     # 4-byte Folded Spill
; CHECK-LABEL: .LBB0_2:
; CHECK:	chsi	160(%r15), 2            # 4-byte Folded Reload

entry:
  %cmp  = icmp eq i32 %arg, 0
  br i1 %cmp, label %cond, label %exit

cond:
  %val0 = load i32, ptr %src
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  br label %exit

exit:
  %tmp0 = phi i32 [0, %entry], [%val0, %cond]
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  %cmp0 = icmp ne i32 %tmp0, 2
  %zxt0 = zext i1 %cmp0 to i32
  %and0 = and i32 %arg, %zxt0

  ret i32 %and0
}

define i64 @fun1(ptr %src, i64 %arg) nounwind {
; CHECK-LABEL: fun1:
; CHECK: 	mvghi	160(%r15), 0            # 8-byte Folded Spill
; CHECK:	mvc	160(8,%r15), 0(%r2)     # 8-byte Folded Spill
; CHECK-LABEL: .LBB1_2:
; CHECK:	cghsi	160(%r15), 2            # 8-byte Folded Reload
entry:
  %cmp  = icmp eq i64 %arg, 0
  br i1 %cmp, label %cond, label %exit

cond:
  %val0 = load i64, ptr %src
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  br label %exit

exit:
  %tmp0 = phi i64 [0, %entry], [%val0, %cond]
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  %cmp0 = icmp ne i64 %tmp0, 2
  %zxt0 = zext i1 %cmp0 to i64
  %and0 = and i64 %arg, %zxt0

  ret i64 %and0
}
