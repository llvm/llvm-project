; RUN: llc -mtriple=x86_64-apple-darwin -stop-after finalize-isel <%s | FileCheck %s

; Check that the callee excludes the return register (%rax) from the list of
; callee-saved-registers.
define preserve_allcc i64 @callee1(i64 %a0, i64 %b0, i64 %c0, i64 %d0, i64 %e0) nounwind {
  %a1 = mul i64 %a0, %b0
  %a2 = mul i64 %a1, %c0
  %a3 = mul i64 %a2, %d0
  %a4 = mul i64 %a3, %e0
  ret i64 %a4
}
; CHECK: name: callee1
; CHECK: calleeSavedRegisters: [ '$rbx', '$r12', '$r13', '$r14', '$r15', '$rbp',
; CHECK:                         '$rcx', '$rdx', '$rsi', '$rdi', '$r8', '$r9', '$r10',
; CHECK:                         '$xmm0', '$xmm1', '$xmm2', '$xmm3', '$xmm4', '$xmm5',
; CHECK:                         '$xmm6', '$xmm7', '$xmm8', '$xmm9', '$xmm10', '$xmm11',
; CHECK:                         '$xmm12', '$xmm13', '$xmm14', '$xmm15' ]
; CHECK: RET 0, $rax

; Check that RegMask contains parameter registers (%rdi, %rsi, %rdx, %rcx,
; %r8), but doesn't contain the return register (%rax).
define i64 @caller1(i64 %a0) nounwind {
  %b1 = call preserve_allcc i64 @callee1(i64 %a0, i64 %a0, i64 %a0, i64 %a0, i64 %a0)
  %b2 = add i64 %b1, %a0
  ret i64 %b2
}
; CHECK:    name: caller1
; CHECK:    CALL64pcrel32 @callee1, CustomRegMask($bh,$bl,$bp,$bph,$bpl,$bx,$ch,$cl,$cx,$dh,$di,$dih,$dil,$dl,$dx,$ebp,$ebx,$ecx,$edi,$edx,$esi,$hbp,$hbx,$hcx,$hdi,$hdx,$hsi,$rbp,$rbx,$rcx,$rdi,$rdx,$rsi,$si,$sih,$sil,$r8,$r9,$r10,$r12,$r13,$r14,$r15,$xmm0,$xmm1,$xmm2,$xmm3,$xmm4,$xmm5,$xmm6,$xmm7,$xmm8,$xmm9,$xmm10,$xmm11,$xmm12,$xmm13,$xmm14,$xmm15,$r8b,$r9b,$r10b,$r12b,$r13b,$r14b,$r15b,$r8bh,$r9bh,$r10bh,$r12bh,$r13bh,$r14bh,$r15bh,$r8d,$r9d,$r10d,$r12d,$r13d,$r14d,$r15d,$r8w,$r9w,$r10w,$r12w,$r13w,$r14w,$r15w,$r8wh,$r9wh,$r10wh,$r12wh,$r13wh,$r14wh,$r15wh), implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp, implicit-def $rax
; CHECK:    RET 0, $rax


; Check that the callee excludes the return registers (%rax, %rdx) from the list
; of callee-saved-registers.
define preserve_allcc {i64, i64} @callee2(i64 %a0, i64 %b0, i64 %c0, i64 %d0, i64 %e0) nounwind {
  %a1 = mul i64 %a0, %b0
  %a2 = mul i64 %a1, %c0
  %a3 = mul i64 %a2, %d0
  %a4 = mul i64 %a3, %e0
  %b4 = insertvalue {i64, i64} undef, i64 %a3, 0
  %b5 = insertvalue {i64, i64} %b4, i64 %a4, 1
  ret {i64, i64} %b5
}
; CHECK: name: callee2
; CHECK: calleeSavedRegisters: [ '$rbx', '$r12', '$r13', '$r14', '$r15', '$rbp',
; CHECK:                         '$rcx', '$rsi', '$rdi', '$r8', '$r9', '$r10', '$xmm0',
; CHECK:                         '$xmm1', '$xmm2', '$xmm3', '$xmm4', '$xmm5', '$xmm6',
; CHECK:                         '$xmm7', '$xmm8', '$xmm9', '$xmm10', '$xmm11',
; CHECK:                         '$xmm12', '$xmm13', '$xmm14', '$xmm15' ]
; CHECK: RET 0, $rax, $rdx


; Check that RegMask contains parameter registers (%rdi, %rsi, %rdx, %rcx,
; %r8), but doesn't contain the return registers (%rax, %rdx).
define {i64, i64} @caller2(i64 %a0) nounwind {
  %b1 = call preserve_allcc {i64, i64} @callee2(i64 %a0, i64 %a0, i64 %a0, i64 %a0, i64 %a0)
  ret {i64, i64} %b1
}
; CHECK:    name: caller2
; CHECL:    CALL64pcrel32 @callee2, CustomRegMask($bh,$bl,$bp,$bph,$bpl,$bx,$ch,$cl,$cx,$di,$dih,$dil,$ebp,$ebx,$ecx,$edi,$esi,$hbp,$hbx,$hcx,$hdi,$hsi,$rbp,$rbx,$rcx,$rdi,$rsi,$si,$sih,$sil,$r8,$r9,$r10,$r12,$r13,$r14,$r15,$xmm0,$xmm1,$xmm2,$xmm3,$xmm4,$xmm5,$xmm6,$xmm7,$xmm8,$xmm9,$xmm10,$xmm11,$xmm12,$xmm13,$xmm14,$xmm15,$r8b,$r9b,$r10b,$r12b,$r13b,$r14b,$r15b,$r8bh,$r9bh,$r10bh,$r12bh,$r13bh,$r14bh,$r15bh,$r8d,$r9d,$r10d,$r12d,$r13d,$r14d,$r15d,$r8w,$r9w,$r10w,$r12w,$r13w,$r14w,$r15w,$r8wh,$r9wh,$r10wh,$r12wh,$r13wh,$r14wh,$r15wh), implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp, implicit-def $rax, implicit-def $rdx
; CHECK:    RET 0, $rax, $rdx


%struct.Large = type { i64, double, double }

; Declare the callee with a sret parameter.
declare preserve_allcc void @callee3(ptr noalias nocapture writeonly sret(%struct.Large) align 4 %a0, i64 %b0) nounwind;

; Check that RegMask contains %rax and subregisters.
define void @caller3(i64 %a0) nounwind {
  %a1 = alloca %struct.Large, align 8
  call preserve_allcc void @callee3(ptr nonnull sret(%struct.Large) align 8 %a1, i64 %a0)
  ret void
}
; CHECK:    name: caller3
; CHECK:    CALL64pcrel32 @callee3, CustomRegMask($ah,$al,$ax,$bh,$bl,$bp,$bph,$bpl,$bx,$ch,$cl,$cx,$dh,$di,$dih,$dil,$dl,$dx,$eax,$ebp,$ebx,$ecx,$edi,$edx,$esi,$hax,$hbp,$hbx,$hcx,$hdi,$hdx,$hsi,$rax,$rbp,$rbx,$rcx,$rdi,$rdx,$rsi,$si,$sih,$sil,$r8,$r9,$r10,$r12,$r13,$r14,$r15,$xmm0,$xmm1,$xmm2,$xmm3,$xmm4,$xmm5,$xmm6,$xmm7,$xmm8,$xmm9,$xmm10,$xmm11,$xmm12,$xmm13,$xmm14,$xmm15,$r8b,$r9b,$r10b,$r12b,$r13b,$r14b,$r15b,$r8bh,$r9bh,$r10bh,$r12bh,$r13bh,$r14bh,$r15bh,$r8d,$r9d,$r10d,$r12d,$r13d,$r14d,$r15d,$r8w,$r9w,$r10w,$r12w,$r13w,$r14w,$r15w,$r8wh,$r9wh,$r10wh,$r12wh,$r13wh,$r14wh,$r15wh), implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit-def $rsp, implicit-def $ssp
; CHECK:    RET 0


; Check that the callee excludes the return registers (%rax, %xmm0) from the list
; of callee-saved-registers.
define preserve_allcc {i64, double} @callee4(i64 %a0, i64 %b0, i64 %c0, i64 %d0, i64 %e0) nounwind {
  %a1 = mul i64 %a0, %b0
  %a2 = mul i64 %a1, %c0
  %a3 = mul i64 %a2, %d0
  %a4 = mul i64 %a3, %e0
  %b4 = insertvalue {i64, double} undef, i64 %a3, 0
  %b5 = insertvalue {i64, double} %b4, double 1.2, 1
  ret {i64, double} %b5
}
; CHECK: name: callee4
; CHECK: calleeSavedRegisters: [ '$rbx', '$r12', '$r13', '$r14', '$r15', '$rbp',
; CHECK:                         '$rcx', '$rdx', '$rsi', '$rdi', '$r8', '$r9', '$r10',
; CHECK:                         '$xmm1', '$xmm2', '$xmm3', '$xmm4', '$xmm5', '$xmm6',
; CHECK:                         '$xmm7', '$xmm8', '$xmm9', '$xmm10', '$xmm11',
; CHECK:                         '$xmm12', '$xmm13', '$xmm14', '$xmm15' ]
; CHECK: RET 0, $rax, $xmm0

; Check that RegMask contains parameter registers (%rdi, %rsi, %rdx, %rcx,
; %r8), but doesn't contain the return registers (%rax, %xmm0).
define {i64, double} @caller4(i64 %a0) nounwind {
  %b1 = call preserve_allcc {i64, double} @callee4(i64 %a0, i64 %a0, i64 %a0, i64 %a0, i64 %a0)
  ret {i64, double} %b1
}
; CHECK:    name: caller4
; CHECK:    CALL64pcrel32 @callee4, CustomRegMask($bh,$bl,$bp,$bph,$bpl,$bx,$ch,$cl,$cx,$dh,$di,$dih,$dil,$dl,$dx,$ebp,$ebx,$ecx,$edi,$edx,$esi,$hbp,$hbx,$hcx,$hdi,$hdx,$hsi,$rbp,$rbx,$rcx,$rdi,$rdx,$rsi,$si,$sih,$sil,$r8,$r9,$r10,$r12,$r13,$r14,$r15,$xmm1,$xmm2,$xmm3,$xmm4,$xmm5,$xmm6,$xmm7,$xmm8,$xmm9,$xmm10,$xmm11,$xmm12,$xmm13,$xmm14,$xmm15,$r8b,$r9b,$r10b,$r12b,$r13b,$r14b,$r15b,$r8bh,$r9bh,$r10bh,$r12bh,$r13bh,$r14bh,$r15bh,$r8d,$r9d,$r10d,$r12d,$r13d,$r14d,$r15d,$r8w,$r9w,$r10w,$r12w,$r13w,$r14w,$r15w,$r8wh,$r9wh,$r10wh,$r12wh,$r13wh,$r14wh,$r15wh), implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp, implicit-def $rax, implicit-def $xmm0

; CHECK:    RET 0, $rax, $xmm0
