; The purpose of the test case is to ensure that a spill that happens during
; intermediate calculations for a comparison performed in a GPR spills the
; full register. Some i32 comparisons performed in GPRs use code that uses
; the full 64-bits of the register in intermediate stages. Spilling such a value
; as a 32-bit value is incorrect.
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
@glob = local_unnamed_addr global i64 0, align 8
@.str = private unnamed_addr constant [12 x i8] c"Value = %d\0A\00", align 1

; Function Attrs: noinline nounwind
define void @call(i64 %a) local_unnamed_addr #0 {
entry:
  store i64 %a, i64* @glob, align 8
  tail call void asm sideeffect "#Do Nothing", "~{memory}"()
  ret void
}

; Function Attrs: noinline nounwind
define signext i32 @test(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %b, %a
  %sub = sub nsw i32 %add, %c
  %conv = sext i32 %sub to i64
  tail call void @call(i64 %conv)
  tail call void asm sideeffect "#Do Nothing", "~{r0},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"()
  %cmp = icmp sle i32 %add, %c
  %conv1 = zext i1 %cmp to i32
  ret i32 %conv1
; CHECK-LABEL: test
; CHECK: sub r3,
; CHECK: extsw r3,
; CHECK: bl call
; CHECK: sub r3,
; CHECK: rldicl r3, r3, 1, 63
; CHECK: std r3, [[OFF:[0-9]+]](r1)
; CHECK: #APP
; CHECK: ld r3, [[OFF]](r1)
; CHECK: xori r3, r3, 1
; CHECK: blr
}

; Function Attrs: nounwind
define signext i32 @main() local_unnamed_addr #1 {
entry:
  %call = tail call signext i32 @test(i32 signext 10, i32 signext -15, i32 signext 0)
  %call1 = tail call signext i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), i32 signext %call)
  ret i32 0
}

; Function Attrs: nounwind
declare signext i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2
