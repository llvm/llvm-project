; RUN: llc -ppc-gpr-icmps=all -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -ppc-gpr-icmps=all -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -verify-machineinstrs -mcpu=pwr7 -ppc-gen-isel=false < %s | \
; RUN:     FileCheck --check-prefix=CHECK-NO-ISEL %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:     -ppc-asm-full-reg-names -mcpu=pwr10 -ppc-gpr-icmps=none < %s | \
; RUN:     FileCheck %s --check-prefix=CHECK-P10
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:     -ppc-asm-full-reg-names -mcpu=pwr10 -ppc-gpr-icmps=none < %s | \
; RUN:     FileCheck %s --check-prefix=CHECK-P10

; Function Attrs: nounwind readnone
define zeroext i1 @test1(float %v1, float %v2) #0 {
entry:
  %cmp = fcmp oge float %v1, %v2
  %cmp2 = fcmp ole float %v2, 0.000000e+00
  %and5 = and i1 %cmp, %cmp2
  ret i1 %and5

; CHECK-LABEL: @test1
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: li [[REG1:[0-9]+]], 1
; CHECK-DAG: xxlxor [[REG2:[0-9]+]], [[REG2]], [[REG2]]
; CHECK-DAG: fcmpu {{[0-9]+}}, 2, [[REG2]]
; CHECK: crnor
; CHECK: crnor
; CHECK: crnand [[REG4:[0-9]+]],
; CHECK: isel 3, 0, [[REG1]], [[REG4]]
; CHECK-NO-ISEL-LABEL: @test1
; CHECK-NO-ISEL: bc 12, 20, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL-NEXT: blr
; CHECK-NO-ISEL-NEXT: [[TRUE]]
; CHECK-NO-ISEL-NEXT: li 3, 0
; CHECK-NO-ISEL-NEXT: blr
; CHECK: blr

; CHECK-P10-LABEL: test1:
; CHECK-P10:       # %bb.0: # %entry
; CHECK-P10-NEXT:    fcmpu cr0, f1, f2
; CHECK-P10-NEXT:    xxlxor f0, f0, f0
; CHECK-P10-NEXT:    fcmpu cr1, f2, f2
; CHECK-P10-NEXT:    crnor 4*cr5+lt, un, lt
; CHECK-P10-NEXT:    fcmpu cr0, f2, f0
; CHECK-P10-NEXT:    crnor 4*cr5+gt, 4*cr1+un, gt
; CHECK-P10-NEXT:    crand 4*cr5+lt, 4*cr5+lt, 4*cr5+gt
; CHECK-P10-NEXT:    setbc r3, 4*cr5+lt
; CHECK-P10-NEXT:    blr
}

; Function Attrs: nounwind readnone
define zeroext i1 @test2(float %v1, float %v2) #0 {
entry:
  %cmp = fcmp oge float %v1, %v2
  %cmp2 = fcmp ole float %v2, 0.000000e+00
  %xor5 = xor i1 %cmp, %cmp2
  ret i1 %xor5

; CHECK-LABEL: @test2
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: li [[REG1:[0-9]+]], 1
; CHECK-DAG: xxlxor [[REG2:[0-9]+]], [[REG2]], [[REG2]]
; CHECK-DAG: fcmpu {{[0-9]+}}, 2, [[REG2]]
; CHECK: crnor
; CHECK: crnor
; CHECK: creqv [[REG4:[0-9]+]],
; CHECK: isel 3, 0, [[REG1]], [[REG4]]
; CHECK: blr

; CHECK-P10-LABEL: test2:
; CHECK-P10:       # %bb.0: # %entry
; CHECK-P10-NEXT:    fcmpu cr0, f1, f2
; CHECK-P10-NEXT:    xxlxor f0, f0, f0
; CHECK-P10-NEXT:    fcmpu cr1, f2, f2
; CHECK-P10-NEXT:    crnor 4*cr5+lt, un, lt
; CHECK-P10-NEXT:    fcmpu cr0, f2, f0
; CHECK-P10-NEXT:    crnor 4*cr5+gt, 4*cr1+un, gt
; CHECK-P10-NEXT:    crxor 4*cr5+lt, 4*cr5+lt, 4*cr5+gt
; CHECK-P10-NEXT:    setbc r3, 4*cr5+lt
; CHECK-P10-NEXT:    blr
}

; Function Attrs: nounwind readnone
define zeroext i1 @test3(float %v1, float %v2, i32 signext %x) #0 {
entry:
  %cmp = fcmp oge float %v1, %v2
  %cmp2 = fcmp ole float %v2, 0.000000e+00
  %cmp4 = icmp ne i32 %x, -2
  %and7 = and i1 %cmp2, %cmp4
  %xor8 = xor i1 %cmp, %and7
  ret i1 %xor8

; CHECK-LABEL: @test3
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: li [[REG1:[0-9]+]], 1
; CHECK-DAG: xxlxor [[REG2:[0-9]+]], [[REG2]], [[REG2]]
; CHECK-DAG: fcmpu {{[0-9]+}}, 2, [[REG2]]
; CHECK: crnor
; CHECK: crnor
; CHECK: crandc
; CHECK: creqv [[REG4:[0-9]+]],
; CHECK: isel 3, 0, [[REG1]], [[REG4]]
; CHECK: blr

; CHECK-P10-LABEL: test3:
; CHECK-P10:       # %bb.0: # %entry
; CHECK-P10-NEXT:    fcmpu cr0, f1, f2
; CHECK-P10-NEXT:    xxlxor f0, f0, f0
; CHECK-P10-NEXT:    fcmpu cr1, f2, f2
; CHECK-P10-NEXT:    crnor 4*cr5+lt, un, lt
; CHECK-P10-NEXT:    fcmpu cr0, f2, f0
; CHECK-P10-NEXT:    crnor 4*cr5+gt, 4*cr1+un, gt
; CHECK-P10-NEXT:    cmpwi r5, -2
; CHECK-P10-NEXT:    crandc 4*cr5+gt, 4*cr5+gt, eq
; CHECK-P10-NEXT:    crxor 4*cr5+lt, 4*cr5+lt, 4*cr5+gt
; CHECK-P10-NEXT:    setbc r3, 4*cr5+lt
; CHECK-P10-NEXT:    blr
}

; Function Attrs: nounwind readnone
define zeroext i1 @test4(i1 zeroext %v1, i1 zeroext %v2, i1 zeroext %v3) #0 {
entry:
  %and8 = and i1 %v1, %v2
  %or9 = or i1 %and8, %v3
  ret i1 %or9

; CHECK-DAG: @test4
; CHECK: and [[REG1:[0-9]+]], 3, 4
; CHECK: or 3, [[REG1]], 5
; CHECK: blr
}

; Function Attrs: nounwind readnone
define zeroext i1 @test5(i1 zeroext %v1, i1 zeroext %v2, i32 signext %v3) #0 {
entry:
  %and6 = and i1 %v1, %v2
  %cmp = icmp ne i32 %v3, -2
  %or7 = or i1 %and6, %cmp
  ret i1 %or7

; CHECK-LABEL: @test5
; CHECK-DAG: li [[NEG2:[0-9]+]], -2
; CHECK-DAG: and [[REG1:[0-9]+]], 3, 4
; CHECK-DAG: xor [[NE1:[0-9]+]], 5, [[NEG2]]
; CHECK-DAG: clrldi [[TRUNC:[0-9]+]], [[REG1]], 63
; CHECK-DAG: cntlzw [[NE2:[0-9]+]], [[NE1]]
; CHECK: srwi [[NE3:[0-9]+]], [[NE2]], 5
; CHECK: xori [[NE4:[0-9]+]], [[NE3]], 1
; CHECK: or 3, [[TRUNC]], [[NE4]]
; CHECK-NEXT: blr

; CHECK-P10-LABEL: test5:
; CHECK-P10:       # %bb.0: # %entry
; CHECK-P10-NEXT:    and r3, r3, r4
; CHECK-P10-NEXT:    cmpwi cr1, r5, -2
; CHECK-P10-NEXT:    andi. r3, r3, 1
; CHECK-P10-NEXT:    crorc 4*cr5+lt, gt, 4*cr1+eq
; CHECK-P10-NEXT:    setbc r3, 4*cr5+lt
; CHECK-P10-NEXT:    blr
}

; Function Attrs: nounwind readnone
define zeroext i1 @test6(i1 zeroext %v1, i1 zeroext %v2, i32 signext %v3) #0 {
entry:
  %cmp = icmp ne i32 %v3, -2
  %or6 = or i1 %cmp, %v2
  %and7 = and i1 %or6, %v1
  ret i1 %and7

; CHECK-LABEL: @test6
; CHECK-DAG: li [[NEG2:[0-9]+]], -2
; CHECK-DAG: clrldi [[CLR1:[0-9]+]], 4, 63
; CHECK-DAG: clrldi [[CLR2:[0-9]+]], 3, 63
; CHECK-DAG: xor [[NE1:[0-9]+]], 5, [[NEG2]]
; CHECK-DAG: cntlzw [[NE2:[0-9]+]], [[NE1]]
; CHECK: srwi [[NE3:[0-9]+]], [[NE2]], 5
; CHECK: xori [[NE4:[0-9]+]], [[NE3]], 1
; CHECK: or [[OR:[0-9]+]], [[NE4]], [[CLR1]]
; CHECK: and 3, [[OR]], [[CLR2]]
; CHECK-NEXT: blr

; CHECK-P10-LABEL: test6:
; CHECK-P10:       # %bb.0: # %entry
; CHECK-P10-NEXT:    andi. r3, r3, 1
; CHECK-P10-NEXT:    cmpwi cr1, r5, -2
; CHECK-P10-NEXT:    crmove 4*cr5+lt, gt
; CHECK-P10-NEXT:    andi. r3, r4, 1
; CHECK-P10-NEXT:    crorc 4*cr5+gt, gt, 4*cr1+eq
; CHECK-P10-NEXT:    crand 4*cr5+lt, 4*cr5+gt, 4*cr5+lt
; CHECK-P10-NEXT:    setbc r3, 4*cr5+lt
; CHECK-P10-NEXT:    blr
}

; Function Attrs: nounwind readnone
define signext i32 @test7(i1 zeroext %v2, i32 signext %i1, i32 signext %i2) #0 {
entry:
  %cond = select i1 %v2, i32 %i1, i32 %i2
  ret i32 %cond

; CHECK-LABEL: @test7
; CHECK: andi. {{[0-9]+}}, 3, 1
; CHECK: iselgt 3, 4, 5
; CHECK: blr
}

define signext i32 @exttest7(i32 signext %a) #0 {
entry:
  %cmp = icmp eq i32 %a, 5
  %cond = select i1 %cmp, i32 7, i32 8
  ret i32 %cond

; CHECK-LABEL: @exttest7
; CHECK-DAG: cmpwi 3, 5
; CHECK-DAG: li [[REG1:[0-9]+]], 8
; CHECK-DAG: li [[REG2:[0-9]+]], 7
; CHECK: iseleq 3, [[REG2]], [[REG1]]
; CHECK-NOT: rldicl
; CHECK: blr
}

define zeroext i32 @exttest8() #0 {
entry:
  %v0 = load i64, i64* undef, align 8
  %sub = sub i64 80, %v0
  %div = lshr i64 %sub, 1
  %conv13 = trunc i64 %div to i32
  %cmp14 = icmp ugt i32 %conv13, 80
  %.conv13 = select i1 %cmp14, i32 0, i32 %conv13
  ret i32 %.conv13
; CHECK-LABEL: @exttest8
; This is a don't-crash test: %conv13 is both one of the possible select output
; values and also an input to the conditional feeding it.
}

; Function Attrs: nounwind readnone
define float @test8(i1 zeroext %v2, float %v1, float %v3) #0 {
entry:
  %cond = select i1 %v2, float %v1, float %v3
  ret float %cond

; CHECK-LABEL: @test8
; CHECK: andi. {{[0-9]+}}, 3, 1
; CHECK: bclr 12, 1, 0
; CHECK: fmr 1, 2
; CHECK: blr
}

; Function Attrs: nounwind readnone
define signext i32 @test10(i32 signext %v1, i32 signext %v2) #0 {
entry:
  %tobool = icmp ne i32 %v1, 0
  %lnot = icmp eq i32 %v2, 0
  %and3 = and i1 %tobool, %lnot
  %and = zext i1 %and3 to i32
  ret i32 %and

; CHECK-LABEL: @test10
; CHECK-DAG: cntlzw 3, 3
; CHECK-DAG: cntlzw 4, 4
; CHECK-DAG: srwi 3, 3, 5
; CHECK-DAG: srwi 4, 4, 5
; CHECK: xori 3, 3, 1
; CHECK: and 3, 3, 4
; CHECK-NEXT: blr

; CHECK-P10-LABEL: test10:
; CHECK-P10:       # %bb.0: # %entry
; CHECK-P10-NEXT:    cmpwi r3, 0
; CHECK-P10-NEXT:    cmpwi cr1, r4, 0
; CHECK-P10-NEXT:    crandc 4*cr5+lt, 4*cr1+eq, eq
; CHECK-P10-NEXT:    setbc r3, 4*cr5+lt
; CHECK-P10-NEXT:    blr
}

attributes #0 = { nounwind readnone }

