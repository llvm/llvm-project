; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-S,CHECK-ALL
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-S,CHECK-ALL
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-P9,CHECK-ALL

@globalVar = common dso_local local_unnamed_addr global i32 0, align 4
@externGlobalVar = external local_unnamed_addr global i32, align 4
@indirectCall = common dso_local local_unnamed_addr global i32 (i32)* null, align 8

; This funcion needs to remain as noinline.
; The compiler needs to know this function is local but must be forced to call
; it. The only thing we really need to check here is that st_other=0 and
; so we make sure that there is no .localentry.
define dso_local signext i32 @localCall(i32 signext %a) local_unnamed_addr #0 {
; CHECK-ALL-LABEL: localCall:
; CHECK-S-NOT:   .localentry
; CHECK-S:         addi r3, r3, 5
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %a, 5
  ret i32 %add
}

define dso_local signext i32 @DirectCallLocal1(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: DirectCallLocal1:
; CHECK-S:         .localentry     DirectCallLocal1
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    bl localCall@notoc
; CHECK-S-NEXT:    plwz r4, globalVar@PCREL(0), 1
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %call = tail call signext i32 @localCall(i32 signext %add)
  %0 = load i32, i32* @globalVar, align 4
  %mul = mul nsw i32 %0, %call
  ret i32 %mul
}

define dso_local signext i32 @DirectCallLocal2(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: DirectCallLocal2:
; CHECK-S:         .localentry     DirectCallLocal2
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    bl localCall@notoc
; CHECK-S-NEXT:    pld r4, externGlobalVar@got@pcrel(0), 1
; CHECK-S-NEXT: .Lpcrel0:
; CHECK-S-NEXT:    .reloc .Lpcrel0-8,R_PPC64_PCREL_OPT,.-(.Lpcrel0-8)
; CHECK-S-NEXT:    lwz r4, 0(r4)
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %call = tail call signext i32 @localCall(i32 signext %add)
  %0 = load i32, i32* @externGlobalVar, align 4
  %mul = mul nsw i32 %0, %call
  ret i32 %mul
}

define dso_local signext i32 @DirectCallLocalNoGlobal(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: DirectCallLocalNoGlobal:
; CHECK-S:         .localentry DirectCallLocalNoGlobal, 1
; CHECK-S-NEXT:    # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    .cfi_def_cfa_offset 48
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    .cfi_offset r30, -16
; CHECK-S-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -48(r1)
; CHECK-S-NEXT:    mr r30, r4
; CHECK-S-NEXT:    bl localCall@notoc
; CHECK-S-NEXT:    add r3, r3, r30
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 48
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %call = tail call signext i32 @localCall(i32 signext %a)
  %add = add nsw i32 %call, %b
  ret i32 %add
}

define dso_local signext i32 @DirectCallExtern1(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: DirectCallExtern1:
; CHECK-S:         .localentry     DirectCallExtern1
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    bl externCall@notoc
; CHECK-S-NEXT:    plwz r4, globalVar@PCREL(0), 1
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %call = tail call signext i32 @externCall(i32 signext %add)
  %0 = load i32, i32* @globalVar, align 4
  %mul = mul nsw i32 %0, %call
  ret i32 %mul
}

declare signext i32 @externCall(i32 signext) local_unnamed_addr

define dso_local signext i32 @DirectCallExtern2(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: DirectCallExtern2:
; CHECK-S:         .localentry     DirectCallExtern2
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    bl externCall@notoc
; CHECK-S-NEXT:    pld r4, externGlobalVar@got@pcrel(0), 1
; CHECK-S-NEXT:  .Lpcrel1:
; CHECK-S-NEXT:    .reloc .Lpcrel1-8,R_PPC64_PCREL_OPT,.-(.Lpcrel1-8)
; CHECK-S-NEXT:    lwz r4, 0(r4)
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %call = tail call signext i32 @externCall(i32 signext %add)
  %0 = load i32, i32* @externGlobalVar, align 4
  %mul = mul nsw i32 %0, %call
  ret i32 %mul
}

define dso_local signext i32 @DirectCallExternNoGlobal(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: DirectCallExternNoGlobal:
; CHECK-S:         .localentry DirectCallExternNoGlobal, 1
; CHECK-P9:        .localentry DirectCallExternNoGlobal, .Lfunc_lep6-.Lfunc_gep6
; CHECK-ALL:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    .cfi_def_cfa_offset 48
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    .cfi_offset r30, -16
; CHECK-S-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -48(r1)
; CHECK-S-NEXT:    mr r30, r4
; CHECK-S-NEXT:    bl externCall@notoc
; CHECK-S-NEXT:    add r3, r3, r30
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 48
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %call = tail call signext i32 @externCall(i32 signext %a)
  %add = add nsw i32 %call, %b
  ret i32 %add
}

define dso_local signext i32 @TailCallLocal1(i32 signext %a) local_unnamed_addr {
; CHECK-ALL-LABEL: TailCallLocal1:
; CHECK-S:         .localentry     TailCallLocal1
; CHECK-S:       # %bb.0: # %entry
; CHECK-S:         plwz r4, globalVar@PCREL(0), 1
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    b localCall@notoc
entry:
  %0 = load i32, i32* @globalVar, align 4
  %add = add nsw i32 %0, %a
  %call = tail call signext i32 @localCall(i32 signext %add)
  ret i32 %call
}

define dso_local signext i32 @TailCallLocal2(i32 signext %a) local_unnamed_addr {
; CHECK-ALL-LABEL: TailCallLocal2:
; CHECK-S:         .localentry     TailCallLocal2
; CHECK-S:       # %bb.0: # %entry
; CHECK-S:         pld r4, externGlobalVar@got@pcrel(0), 1
; CHECK-S-NEXT:  .Lpcrel2:
; CHECK-S-NEXT:    .reloc .Lpcrel2-8,R_PPC64_PCREL_OPT,.-(.Lpcrel2-8)
; CHECK-S-NEXT:    lwz r4, 0(r4)
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    b localCall@notoc
entry:
  %0 = load i32, i32* @externGlobalVar, align 4
  %add = add nsw i32 %0, %a
  %call = tail call signext i32 @localCall(i32 signext %add)
  ret i32 %call
}

define dso_local signext i32 @TailCallLocalNoGlobal(i32 signext %a) local_unnamed_addr {
; CHECK-ALL-LABEL: TailCallLocalNoGlobal:
; CHECK-S:         .localentry TailCallLocalNoGlobal, 1
; CHECK-P9:        .localentry TailCallLocalNoGlobal, .Lfunc_lep9-.Lfunc_gep9
; CHECK-ALL:       # %bb.0: # %entry
; CHECK-S:         b localCall@notoc
entry:
  %call = tail call signext i32 @localCall(i32 signext %a)
  ret i32 %call
}

define dso_local signext i32 @TailCallExtern1(i32 signext %a) local_unnamed_addr {
; CHECK-ALL-LABEL: TailCallExtern1:
; CHECK-S:         .localentry     TailCallExtern1
; CHECK-S:       # %bb.0: # %entry
; CHECK-S:         plwz r4, globalVar@PCREL(0), 1
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    b externCall@notoc
entry:
  %0 = load i32, i32* @globalVar, align 4
  %add = add nsw i32 %0, %a
  %call = tail call signext i32 @externCall(i32 signext %add)
  ret i32 %call
}

define dso_local signext i32 @TailCallExtern2(i32 signext %a) local_unnamed_addr {
; CHECK-ALL-LABEL: TailCallExtern2:
; CHECK-S:         .localentry     TailCallExtern2
; CHECK-S:       # %bb.0: # %entry
; CHECK-S:         pld r4, externGlobalVar@got@pcrel(0), 1
; CHECK-S-NEXT:  .Lpcrel3:
; CHECK-S-NEXT:    .reloc .Lpcrel3-8,R_PPC64_PCREL_OPT,.-(.Lpcrel3-8)
; CHECK-S-NEXT:    lwz r4, 0(r4)
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    b externCall@notoc
entry:
  %0 = load i32, i32* @externGlobalVar, align 4
  %add = add nsw i32 %0, %a
  %call = tail call signext i32 @externCall(i32 signext %add)
  ret i32 %call
}

define dso_local signext i32 @TailCallExternNoGlobal(i32 signext %a) local_unnamed_addr {
; CHECK-ALL-LABEL: TailCallExternNoGlobal:
; CHECK-S:         .localentry TailCallExternNoGlobal, 1
; CHECK-S-NEXT:  # %bb.0: # %entry
; CHECK-S-NEXT:    b externCall@notoc
; CHECK-S-NEXT:    #TC_RETURNd8 externCall@notoc
entry:
  %call = tail call signext i32 @externCall(i32 signext %a)
  ret i32 %call
}

define dso_local signext i32 @IndirectCall1(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: IndirectCall1:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    pld r12, indirectCall@PCREL(0), 1
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    mtctr r12
; CHECK-S-NEXT:    bctrl
; CHECK-S-NEXT:    plwz r4, globalVar@PCREL(0), 1
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %0 = load i32 (i32)*, i32 (i32)** @indirectCall, align 8
  %call = tail call signext i32 %0(i32 signext %add)
  %1 = load i32, i32* @globalVar, align 4
  %mul = mul nsw i32 %1, %call
  ret i32 %mul
}

define dso_local signext i32 @IndirectCall2(i32 signext %a, i32 signext %b) local_unnamed_addr {
; CHECK-ALL-LABEL: IndirectCall2:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    pld r12, indirectCall@PCREL(0), 1
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    mtctr r12
; CHECK-S-NEXT:    bctrl
; CHECK-S-NEXT:    pld r4, externGlobalVar@got@pcrel(0), 1
; CHECK-S-NEXT:  .Lpcrel4:
; CHECK-S-NEXT:    .reloc .Lpcrel4-8,R_PPC64_PCREL_OPT,.-(.Lpcrel4-8)
; CHECK-S-NEXT:    lwz r4, 0(r4)
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %0 = load i32 (i32)*, i32 (i32)** @indirectCall, align 8
  %call = tail call signext i32 %0(i32 signext %add)
  %1 = load i32, i32* @externGlobalVar, align 4
  %mul = mul nsw i32 %1, %call
  ret i32 %mul
}

define dso_local signext i32 @IndirectCall3(i32 signext %a, i32 signext %b, i32 (i32)* nocapture %call_param) local_unnamed_addr {
; CHECK-ALL-LABEL: IndirectCall3:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -32(r1)
; CHECK-S-NEXT:    .cfi_def_cfa_offset 32
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    add r3, r4, r3
; CHECK-S-NEXT:    mr r12, r5
; CHECK-S-NEXT:    mtctr r5
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    bctrl
; CHECK-S-NEXT:    plwz r4, globalVar@PCREL(0), 1
; CHECK-S-NEXT:    mullw r3, r4, r3
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 32
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %add = add nsw i32 %b, %a
  %call = tail call signext i32 %call_param(i32 signext %add)
  %0 = load i32, i32* @globalVar, align 4
  %mul = mul nsw i32 %0, %call
  ret i32 %mul
}

define dso_local signext i32 @IndirectCallNoGlobal(i32 signext %a, i32 signext %b, i32 (i32)* nocapture %call_param) local_unnamed_addr {
; CHECK-ALL-LABEL: IndirectCallNoGlobal:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mflr r0
; CHECK-S-NEXT:    .cfi_def_cfa_offset 48
; CHECK-S-NEXT:    .cfi_offset lr, 16
; CHECK-S-NEXT:    .cfi_offset r30, -16
; CHECK-S-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; CHECK-S-NEXT:    std r0, 16(r1)
; CHECK-S-NEXT:    stdu r1, -48(r1)
; CHECK-S-NEXT:    mr r12, r5
; CHECK-S-NEXT:    mtctr r5
; CHECK-S-NEXT:    mr r30, r4
; CHECK-S-NEXT:    bctrl
; CHECK-S-NEXT:    add r3, r3, r30
; CHECK-S-NEXT:    extsw r3, r3
; CHECK-S-NEXT:    addi r1, r1, 48
; CHECK-S-NEXT:    ld r0, 16(r1)
; CHECK-S-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; CHECK-S-NEXT:    mtlr r0
; CHECK-S-NEXT:    blr
entry:
  %call = tail call signext i32 %call_param(i32 signext %a)
  %add = add nsw i32 %call, %b
  ret i32 %add
}

define dso_local signext i32 @IndirectCallOnly(i32 signext %a, i32 (i32)* nocapture %call_param) local_unnamed_addr {
; CHECK-ALL-LABEL: IndirectCallOnly:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    mtctr r4
; CHECK-S-NEXT:    mr r12, r4
; CHECK-S-NEXT:    bctr
; CHECK-S-NEXT:    #TC_RETURNr8 ctr
entry:
  %call = tail call signext i32 %call_param(i32 signext %a)
  ret i32 %call
}

attributes #0 = { noinline }

