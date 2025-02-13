; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:     --check-prefix=CHECK64
; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:     --check-prefix=CHECK32
; RUN: llc -verify-machineinstrs -mtriple powerpc64le-unknown-linux \
; RUN:     -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:     --check-prefix=LINUX64LE
; RUN: llc -verify-machineinstrs -mtriple powerpc64-unknown-linux \
; RUN:     -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:     --check-prefix=LINUX64BE

@.str = private unnamed_addr constant [47 x i8] c"TLS variable 1, 2 and non-TLS var: %s, %s, %s\0A\00", align 1
@a = internal thread_local constant [5 x i8] c"tls1\00", align 1
@b = internal thread_local constant [5 x i8] c"tls2\00", align 1
@c = internal constant [15 x i8] c"Regular global\00", align 1
@d = internal constant [10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10], align 4
@e = internal constant [4 x float] [float 0x4055F33340000000, float 0x4056333340000000, float 0x40567999A0000000, float 0x4056B33340000000], align 4

declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #1
declare void @callee(ptr noundef) local_unnamed_addr #3
declare void @callee2(ptr noundef) local_unnamed_addr #3

define void @print_tls_func() {
; CHECK64-LABEL: print_tls_func:
; CHECK64:       # %bb.0: # %entry
; CHECK64-NEXT:    mflr r0
; CHECK64-NEXT:    stdu r1, -112(r1)
; CHECK64-NEXT:    ld r3, L..C0(r2) # target-flags(ppc-tlsldm) @"_$TLSML"
; CHECK64-NEXT:    std r0, 128(r1)
; CHECK64-NEXT:    ld r6, L..C1(r2) # @_MergedGlobals
; CHECK64-NEXT:    bla .__tls_get_mod[PR]
; CHECK64-NEXT:    ld r4, L..C2(r2) # target-flags(ppc-tlsld) @a
; CHECK64-NEXT:    ld r5, L..C3(r2) # target-flags(ppc-tlsld) @b
; CHECK64-NEXT:    add r4, r3, r4
; CHECK64-NEXT:    add r5, r3, r5
; CHECK64-NEXT:    addi r3, r6, 72
; CHECK64-NEXT:    bl .printf[PR]
; CHECK64-NEXT:    nop
; CHECK64-NEXT:    addi r1, r1, 112
; CHECK64-NEXT:    ld r0, 16(r1)
; CHECK64-NEXT:    mtlr r0
; CHECK64-NEXT:    blr
;
; CHECK32-LABEL: print_tls_func:
; CHECK32:       # %bb.0: # %entry
; CHECK32-NEXT:    mflr r0
; CHECK32-NEXT:    stwu r1, -64(r1)
; CHECK32-NEXT:    lwz r3, L..C0(r2) # target-flags(ppc-tlsldm) @"_$TLSML"
; CHECK32-NEXT:    stw r0, 72(r1)
; CHECK32-NEXT:    lwz r6, L..C1(r2) # @_MergedGlobals
; CHECK32-NEXT:    bla .__tls_get_mod[PR]
; CHECK32-NEXT:    lwz r4, L..C2(r2) # target-flags(ppc-tlsld) @a
; CHECK32-NEXT:    lwz r5, L..C3(r2) # target-flags(ppc-tlsld) @b
; CHECK32-NEXT:    add r4, r3, r4
; CHECK32-NEXT:    add r5, r3, r5
; CHECK32-NEXT:    addi r3, r6, 72
; CHECK32-NEXT:    bl .printf[PR]
; CHECK32-NEXT:    nop
; CHECK32-NEXT:    addi r1, r1, 64
; CHECK32-NEXT:    lwz r0, 8(r1)
; CHECK32-NEXT:    mtlr r0
; CHECK32-NEXT:    blr
;
; LINUX64LE-LABEL: print_tls_func:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -96(r1)
; LINUX64LE-NEXT:    std r0, 112(r1)
; LINUX64LE-NEXT:    .cfi_def_cfa_offset 96
; LINUX64LE-NEXT:    .cfi_offset lr, 16
; LINUX64LE-NEXT:    addis r3, r13, a@tprel@ha
; LINUX64LE-NEXT:    addi r4, r3, a@tprel@l
; LINUX64LE-NEXT:    addis r3, r13, b@tprel@ha
; LINUX64LE-NEXT:    addi r5, r3, b@tprel@l
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    addi r6, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r6, 72
; LINUX64LE-NEXT:    bl printf
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 96
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
;
; LINUX64BE-LABEL: print_tls_func:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -128(r1)
; LINUX64BE-NEXT:    std r0, 144(r1)
; LINUX64BE-NEXT:    .cfi_def_cfa_offset 128
; LINUX64BE-NEXT:    .cfi_offset lr, 16
; LINUX64BE-NEXT:    .cfi_offset r30, -16
; LINUX64BE-NEXT:    addis r3, r2, a@got@tlsld@ha
; LINUX64BE-NEXT:    std r30, 112(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    addi r3, r3, a@got@tlsld@l
; LINUX64BE-NEXT:    bl __tls_get_addr(a@tlsld)
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addis r4, r2, b@got@tlsld@ha
; LINUX64BE-NEXT:    addis r3, r3, a@dtprel@ha
; LINUX64BE-NEXT:    addi r30, r3, a@dtprel@l
; LINUX64BE-NEXT:    addi r3, r4, b@got@tlsld@l
; LINUX64BE-NEXT:    bl __tls_get_addr(b@tlsld)
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addis r3, r3, b@dtprel@ha
; LINUX64BE-NEXT:    mr r4, r30
; LINUX64BE-NEXT:    addi r5, r3, b@dtprel@l
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    addi r6, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r6, 72
; LINUX64BE-NEXT:    bl printf
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    ld r30, 112(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    addi r1, r1, 128
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
entry:
  %0 = tail call align 1 ptr @llvm.threadlocal.address.p0(ptr align 1 @a)
  %1 = tail call align 1 ptr @llvm.threadlocal.address.p0(ptr align 1 @b)
  %call = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef nonnull @c)
  ret void
}

define void @test_func() {
; CHECK64-LABEL: test_func:
; CHECK64:       # %bb.0: # %entry
; CHECK64-NEXT:    mflr r0
; CHECK64-NEXT:    stdu r1, -112(r1)
; CHECK64-NEXT:    ld r3, L..C1(r2) # @_MergedGlobals
; CHECK64-NEXT:    std r0, 128(r1)
; CHECK64-NEXT:    addi r3, r3, 32
; CHECK64-NEXT:    bl .callee[PR]
; CHECK64-NEXT:    nop
; CHECK64-NEXT:    addi r1, r1, 112
; CHECK64-NEXT:    ld r0, 16(r1)
; CHECK64-NEXT:    mtlr r0
; CHECK64-NEXT:    blr
;
; CHECK32-LABEL: test_func:
; CHECK32:       # %bb.0: # %entry
; CHECK32-NEXT:    mflr r0
; CHECK32-NEXT:    stwu r1, -64(r1)
; CHECK32-NEXT:    lwz r3, L..C1(r2) # @_MergedGlobals
; CHECK32-NEXT:    stw r0, 72(r1)
; CHECK32-NEXT:    addi r3, r3, 32
; CHECK32-NEXT:    bl .callee[PR]
; CHECK32-NEXT:    nop
; CHECK32-NEXT:    addi r1, r1, 64
; CHECK32-NEXT:    lwz r0, 8(r1)
; CHECK32-NEXT:    mtlr r0
; CHECK32-NEXT:    blr

; LINUX64LE-LABEL: test_func:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    .cfi_def_cfa_offset 32
; LINUX64LE-NEXT:    .cfi_offset lr, 16
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 32
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
;
; LINUX64BE-LABEL: test_func:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    .cfi_def_cfa_offset 112
; LINUX64BE-NEXT:    .cfi_offset lr, 16
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 32
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
entry:
  tail call void @callee(ptr noundef nonnull @d) #4
  ret void
}

define void @test_func2() {
; CHECK64-LABEL: test_func2:
; CHECK64:       # %bb.0: # %entry
; CHECK64-NEXT:    mflr r0
; CHECK64-NEXT:    stdu r1, -112(r1)
; CHECK64-NEXT:    ld r3, L..C1(r2) # @_MergedGlobals
; CHECK64-NEXT:    std r0, 128(r1)
; CHECK64-NEXT:    addi r3, r3, 16
; CHECK64-NEXT:    bl .callee2[PR]
; CHECK64-NEXT:    nop
; CHECK64-NEXT:    addi r1, r1, 112
; CHECK64-NEXT:    ld r0, 16(r1)
; CHECK64-NEXT:    mtlr r0
; CHECK64-NEXT:    blr
;
; CHECK32-LABEL: test_func2:
; CHECK32:       # %bb.0: # %entry
; CHECK32-NEXT:    mflr r0
; CHECK32-NEXT:    stwu r1, -64(r1)
; CHECK32-NEXT:    lwz r3, L..C1(r2) # @_MergedGlobals
; CHECK32-NEXT:    stw r0, 72(r1)
; CHECK32-NEXT:    addi r3, r3, 16
; CHECK32-NEXT:    bl .callee2[PR]
; CHECK32-NEXT:    nop
; CHECK32-NEXT:    addi r1, r1, 64
; CHECK32-NEXT:    lwz r0, 8(r1)
; CHECK32-NEXT:    mtlr r0
; CHECK32-NEXT:    blr
;
; LINUX64LE-LABEL: test_func2:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    .cfi_def_cfa_offset 32
; LINUX64LE-NEXT:    .cfi_offset lr, 16
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 16
; LINUX64LE-NEXT:    bl callee2
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
;
; LINUX64BE-LABEL: test_func2:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    .cfi_def_cfa_offset 112
; LINUX64BE-NEXT:    .cfi_offset lr, 16
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 16
; LINUX64BE-NEXT:    bl callee2
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
entry:
  tail call void @callee2(ptr noundef nonnull @e) #4
  ret void
}

; Check the contents of the TLS data and the _MergedGlobals structure to
; check that TLS data has been skipped during global merge.

; CHECK64: 	.csect a[TL],2
; CHECK64-NEXT:	.lglobl	a[TL]
; CHECK64-NEXT:	.string	"tls1"
; CHECK64:	.csect b[TL],2
; CHECK64-NEXT:	.lglobl	b[TL]
; CHECK64-NEXT:	.string	"tls2"
; CHECK64:	.csect L.._MergedGlobals[RO],2
; CHECK64:	.align	2
; CHECK64-LABEL: c:
; CHECK64:	.string	"Regular global"
; CHECK64-LABEL: e:
; CHECK64:	.vbyte	4, 0x42af999a
; CHECK64-NEXT:	.vbyte	4, 0x42b1999a
; CHECK64-NEXT:	.vbyte	4, 0x42b3cccd
; CHECK64-NEXT:	.vbyte	4, 0x42b5999a
; CHECK64-LABEL: d:
; CHECK64:	.vbyte	4, 1
; CHECK64-NEXT:	.vbyte	4, 2
; CHECK64-NEXT:	.vbyte	4, 3
; CHECK64-NEXT:	.vbyte	4, 4
; CHECK64-NEXT:	.vbyte	4, 5
; CHECK64-NEXT:	.vbyte	4, 6
; CHECK64-NEXT:	.vbyte	4, 7
; CHECK64-NEXT:	.vbyte	4, 8
; CHECK64-NEXT:	.vbyte	4, 9
; CHECK64-NEXT:	.vbyte	4, 10
; CHECK64-LABEL: L...str
; CHECK64:	.byte	'T,'L,'S,' ,'v,'a,'r,'i,'a,'b,'l,'e,' ,'1,',,' ,'2,' ,'a,'n,'d,' ,'n,'o,'n,'-,'T,'L,'S,' ,'v,'a,'r,':,' ,'%,'s,',,' ,'%,'s,',,' ,'%,'s,0012,0000
; CHECK64: L..C1:
; CHECK64-NEXT: .tc L.._MergedGlobals[TC],L.._MergedGlobals[RO]
; CHECK64: L..C2:
; CHECK64-NEXT:	.tc a[TC],a[TL]@ld
; CHECK64: L..C3:
; CHECK64-NEXT:	.tc b[TC],b[TL]@ld

; CHECK32: 	.csect a[TL],2
; CHECK32-NEXT:	.lglobl	a[TL]
; CHECK32-NEXT:	.string	"tls1"
; CHECK32:	.csect b[TL],2
; CHECK32-NEXT:	.lglobl	b[TL]
; CHECK32-NEXT:	.string	"tls2"
; CHECK32:	.csect L.._MergedGlobals[RO],2
; CHECK32:	.align	2
; CHECK32-LABEL: c:
; CHECK32:	.string "Regular global"
; CHECK32-LABEL: e:
; CHECK32:	.vbyte	4, 0x42af999a
; CHECK32-NEXT:	.vbyte	4, 0x42b1999a
; CHECK32-NEXT:	.vbyte	4, 0x42b3cccd
; CHECK32-NEXT:	.vbyte	4, 0x42b5999a
; CHECK32-LABEL: d:
; CHECK32:	.vbyte	4, 1
; CHECK32-NEXT:	.vbyte	4, 2
; CHECK32-NEXT:	.vbyte	4, 3
; CHECK32-NEXT:	.vbyte	4, 4
; CHECK32-NEXT:	.vbyte	4, 5
; CHECK32-NEXT:	.vbyte	4, 6
; CHECK32-NEXT:	.vbyte	4, 7
; CHECK32-NEXT:	.vbyte	4, 8
; CHECK32-NEXT:	.vbyte	4, 9
; CHECK32-NEXT:	.vbyte	4, 10
; CHECK32-LABEL: L...str:
; CHECK32:	.byte	'T,'L,'S,' ,'v,'a,'r,'i,'a,'b,'l,'e,' ,'1,',,' ,'2,' ,'a,'n,'d,' ,'n,'o,'n,'-,'T,'L,'S,' ,'v,'a,'r,':,' ,'%,'s,',,' ,'%,'s,',,' ,'%,'s,0012,0000
; CHECK32: L..C1:
; CHECK32-NEXT: .tc L.._MergedGlobals[TC],L.._MergedGlobals[RO]
; CHECK32: L..C2:
; CHECK32-NEXT:	.tc a[TC],a[TL]@ld
; CHECK32: L..C3:
; CHECK32-NEXT:	.tc b[TC],b[TL]@ld

; LINUX64LE: a:
; LINUX64LE-NEXT:       .asciz  "tls1"
; LINUX64LE-NEXT:       .size   a, 5
; LINUX64LE: b:
; LINUX64LE-NEXT:       .asciz  "tls2"
; LINUX64LE-NEXT:       .size   b, 5
; LINUX64LE: .L_MergedGlobals:
; LINUX64LE-NEXT:       .asciz  "Regular global"
; LINUX64LE-NEXT:       .space  1
; LINUX64LE-NEXT:       .long   0x42af999a
; LINUX64LE-NEXT:       .long   0x42b1999a
; LINUX64LE-NEXT:       .long   0x42b3cccd
; LINUX64LE-NEXT:       .long   0x42b5999a
; LINUX64LE-NEXT:       .long   1
; LINUX64LE-NEXT:       .long   2
; LINUX64LE-NEXT:       .long   3
; LINUX64LE-NEXT:       .long   4
; LINUX64LE-NEXT:       .long   5
; LINUX64LE-NEXT:       .long   6
; LINUX64LE-NEXT:       .long   7
; LINUX64LE-NEXT:       .long   8
; LINUX64LE-NEXT:       .long   9
; LINUX64LE-NEXT:       .long   10
; LINUX64LE-NEXT:       .asciz  "TLS variable 1, 2 and non-TLS var: %s, %s, %s\n"

; LINUX64BE: a:
; LINUX64BE-NEXT:       .asciz  "tls1"
; LINUX64BE-NEXT:       .size   a, 5
; LINUX64BE: b:
; LINUX64BE-NEXT:       .asciz  "tls2"
; LINUX64BE-NEXT:       .size   b, 5
; LINUX64BE: .L_MergedGlobals:
; LINUX64BE-NEXT:       .asciz  "Regular global"
; LINUX64BE-NEXT:       .space  1
; LINUX64BE-NEXT:       .long   0x42af999a
; LINUX64BE-NEXT:       .long   0x42b1999a
; LINUX64BE-NEXT:       .long   0x42b3cccd
; LINUX64BE-NEXT:       .long   0x42b5999a
; LINUX64BE-NEXT:       .long   1
; LINUX64BE-NEXT:       .long   2
; LINUX64BE-NEXT:       .long   3
; LINUX64BE-NEXT:       .long   4
; LINUX64BE-NEXT:       .long   5
; LINUX64BE-NEXT:       .long   6
; LINUX64BE-NEXT:       .long   7
; LINUX64BE-NEXT:       .long   8
; LINUX64BE-NEXT:       .long   9
; LINUX64BE-NEXT:       .long   10
; LINUX64BE-NEXT:       .asciz  "TLS variable 1, 2 and non-TLS var: %s, %s, %s\n"
