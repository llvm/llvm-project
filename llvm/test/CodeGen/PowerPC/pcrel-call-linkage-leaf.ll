; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-S,CHECK-ALL
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-S,CHECK-ALL
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-P9,CHECK-ALL
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 --code-model=large -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-LARGE,CHECK-ALL

@global_int = common dso_local local_unnamed_addr global i32 0, align 4

define dso_local signext i32 @NoTOC() local_unnamed_addr {
; CHECK-ALL-LABEL: NoTOC:
; CHECK-S-NOT:     .localentry
; CHECK-S:         li r3, 42
; CHECK-S-NEXT:    blr
entry:
  ret i32 42
}

define dso_local signext i32 @AsmClobberX2(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: AsmClobberX2:
; CHECK-S:         .localentry AsmClobberX2, 1
; CHECK-S:         add r3, r4, r3
; CHECK-S:         #APP
; CHECK-S-NEXT:    nop
; CHECK-S-NEXT:    #NO_APP
; CHECK-S:         blr
entry:
  %add = add nsw i32 %b, %a
  tail call void asm sideeffect "nop", "~{r2}"()
  ret i32 %add
}

; FIXME: This is actually a test case that shows a bug. On power9 and earlier
;        this test should not compile. On later CPUs (like this test) the @toc
;        should be replaced with @pcrel and we won't need R2 and so the problem
;        goes away.
define dso_local signext i32 @AsmClobberX2WithTOC(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: AsmClobberX2WithTOC:
; CHECK-LARGE:     ld r2, .Lfunc_toc2-.Lfunc_gep2(r12)
; CHECK-LARGE:     add r2, r2, r12
; CHECK-S:         .localentry     AsmClobberX2WithTOC
; CHECK-S:         add r3, r4, r3
; CHECK-S-NEXT:    #APP
; CHECK-S-NEXT:    li r2, 0
; CHECK-S-NEXT:    #NO_APP
; CHECK-S-NEXT:    plwz r4, global_int@PCREL(0), 1
; CHECK-S-NEXT:    add r3, r3, r4
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  tail call void asm sideeffect "li 2, 0", "~{r2}"()
  %0 = load i32, i32* @global_int, align 4
  %add1 = add nsw i32 %add, %0
  ret i32 %add1
}

define dso_local signext i32 @AsmClobberX5(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: AsmClobberX5:
; CHECK-S:         .localentry AsmClobberX5, 1
; CHECK-P9-NOT:    .localentry
; CHECK-ALL:       # %bb.0: # %entry
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    #APP
; CHECK-S-NEXT:    nop
; CHECK-S-NEXT:    #NO_APP
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  tail call void asm sideeffect "nop", "~{r5}"()
  ret i32 %add
}

; Clobber all GPRs except R2.
define dso_local signext i32 @AsmClobberNotR2(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: AsmClobberNotR2:
; CHECK-S:         .localentry AsmClobberNotR2, 1
; CHECK-P9-NOT:    .localentry
; CHECK-S:         add r3, r4, r3
; CHECK-S:         stw r3, -148(r1) # 4-byte Folded Spill
; CHECK-S-NEXT:    #APP
; CHECK-S-NEXT:    nop
; CHECK-S-NEXT:    #NO_APP
; CHECK-S-NEXT:    lwz r3, -148(r1) # 4-byte Folded Reload
; CHECK-S:    blr
entry:
  %add = add nsw i32 %b, %a
  tail call void asm sideeffect "nop", "~{r0},~{r1},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"()
  ret i32 %add
}

; Increase register pressure enough to force the register allocator to
; make use of R2.
define dso_local signext i32 @X2IsCallerSaved(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d, i32 signext %e, i32 signext %f, i32 signext %g, i32 signext %h) local_unnamed_addr {
; CHECK-ALL-LABEL: X2IsCallerSaved:
; CHECK-S:         .localentry X2IsCallerSaved, 1
; CHECK-P9-NOT:    .localentry
; CHECK-ALL:       # %bb.0: # %entry
; CHECK-S-NEXT:    std r29, -24(r1) # 8-byte Folded Spill
; CHECK-S-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; CHECK-S-NEXT:    add r11, r4, r3
; CHECK-S-NEXT:    sub r29, r8, r9
; CHECK-S-NEXT:    add r9, r10, r9
; CHECK-S-NEXT:    sub r10, r10, r3
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    sub r12, r4, r5
; CHECK-S-NEXT:    add r0, r6, r5
; CHECK-S-NEXT:    sub r2, r6, r7
; CHECK-S-NEXT:    add r30, r8, r7
; CHECK-S-NEXT:    mullw r3, r3, r11
; CHECK-S-NEXT:    mullw r3, r3, r5
; CHECK-S-NEXT:    mullw r3, r3, r6
; CHECK-S-NEXT:    mullw r3, r3, r12
; CHECK-S-NEXT:    mullw r3, r3, r0
; CHECK-S-NEXT:    mullw r3, r3, r7
; CHECK-S-NEXT:    mullw r3, r3, r8
; CHECK-S-NEXT:    mullw r3, r3, r2
; CHECK-S-NEXT:    mullw r3, r3, r30
; CHECK-S-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; CHECK-S-NEXT:    mullw r3, r3, r29
; CHECK-S-NEXT:    ld r29, -24(r1) # 8-byte Folded Reload
; CHECK-S-NEXT:    mullw r3, r3, r9
; CHECK-S-NEXT:    mullw r3, r3, r10
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %sub = sub nsw i32 %b, %c
  %add1 = add nsw i32 %d, %c
  %sub2 = sub nsw i32 %d, %e
  %add3 = add nsw i32 %f, %e
  %sub4 = sub nsw i32 %f, %g
  %add5 = add nsw i32 %h, %g
  %sub6 = sub nsw i32 %h, %a
  %mul = mul i32 %b, %a
  %mul7 = mul i32 %mul, %add
  %mul8 = mul i32 %mul7, %c
  %mul9 = mul i32 %mul8, %d
  %mul10 = mul i32 %mul9, %sub
  %mul11 = mul i32 %mul10, %add1
  %mul12 = mul i32 %mul11, %e
  %mul13 = mul i32 %mul12, %f
  %mul14 = mul i32 %mul13, %sub2
  %mul15 = mul i32 %mul14, %add3
  %mul16 = mul i32 %mul15, %sub4
  %mul17 = mul i32 %mul16, %add5
  %mul18 = mul i32 %mul17, %sub6
  ret i32 %mul18
}


define dso_local signext i32 @UsesX2AsTOC() local_unnamed_addr {
; CHECK-ALL-LABEL: UsesX2AsTOC:
; CHECK-LARGE:     ld r2, .Lfunc_toc6-.Lfunc_gep6(r12)
; CHECK-LARGE:     add r2, r2, r12
; CHECK-ALL:       # %bb.0: # %entry
entry:
  %0 = load i32, i32* @global_int, align 4
  ret i32 %0
}


define dso_local double @UsesX2AsConstPoolTOC() local_unnamed_addr {
; CHECK-ALL-LABEL: UsesX2AsConstPoolTOC:
; CHECK-LARGE:     ld r2, .Lfunc_toc7-.Lfunc_gep7(r12)
; CHECK-LARGE:     add r2, r2, r12
; CHECK-S-NOT:       .localentry
; CHECK-ALL:       # %bb.0: # %entry
; CHECK-S-NEXT:    xxsplti32dx vs1, 0, 1078011044
; CHECK-S-NEXT:    xxsplti32dx vs1, 1, -337824948
; CHECK-S-NEXT:    # kill: def $f1 killed $f1 killed $vsl1
; CHECK-S-NEXT:    blr
entry:
  ret double 0x404124A4EBDD334C
}


