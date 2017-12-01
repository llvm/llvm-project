; RUN: llc < %s -asm-verbose=false -mtriple=x86_64-unknown-linux | FileCheck %s --check-prefix=CHECK --check-prefix=CMOV
; RUN: llc < %s -asm-verbose=false -mtriple=i686-unknown-linux | FileCheck %s --check-prefix=CHECK --check-prefix=NOCMOV

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Test 2xCMOV patterns exposed after legalization.
; One way to do that is with (select (fcmp une/oeq)), which gets
; legalized to setp/setne.

; CHECK-LABEL: test_select_fcmp_oeq_i32:

; CMOV-NEXT: ucomiss  %xmm1, %xmm0
; CMOV-NEXT: cmovnel  %esi, %edi
; CMOV-NEXT: cmovpl  %esi, %edi
; CMOV-NEXT: movl  %edi, %eax
; CMOV-NEXT: retq

; NOCMOV-NEXT:  flds  8(%esp)
; NOCMOV-NEXT:  flds  4(%esp)
; NOCMOV-NEXT:  fucompp
; NOCMOV-NEXT:  fnstsw  %ax
; NOCMOV-NEXT:  sahf
; NOCMOV-NEXT:  leal  16(%esp), %eax
; NOCMOV-NEXT:  jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:  jp  [[TBB]]
; NOCMOV-NEXT:  leal  12(%esp), %eax
; NOCMOV-NEXT:[[TBB]]:
; NOCMOV-NEXT:  movl  (%eax), %eax
; NOCMOV-NEXT:  retl
define i32 @test_select_fcmp_oeq_i32(float %a, float %b, i32 %c, i32 %d) #0 {
entry:
  %cmp = fcmp oeq float %a, %b
  %r = select i1 %cmp, i32 %c, i32 %d
  ret i32 %r
}

; CHECK-LABEL: test_select_fcmp_oeq_i64:

; CMOV-NEXT:   ucomiss  %xmm1, %xmm0
; CMOV-NEXT:   cmovneq  %rsi, %rdi
; CMOV-NEXT:   cmovpq  %rsi, %rdi
; CMOV-NEXT:   movq  %rdi, %rax
; CMOV-NEXT:   retq

; NOCMOV-NEXT:   flds  8(%esp)
; NOCMOV-NEXT:   flds  4(%esp)
; NOCMOV-NEXT:   fucompp
; NOCMOV-NEXT:   fnstsw  %ax
; NOCMOV-NEXT:   sahf
; NOCMOV-NEXT:   leal  20(%esp), %ecx
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  12(%esp), %ecx
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   movl  (%ecx), %eax
; NOCMOV-NEXT:   movl  4(%ecx), %edx
; NOCMOV-NEXT:   retl
define i64 @test_select_fcmp_oeq_i64(float %a, float %b, i64 %c, i64 %d) #0 {
entry:
  %cmp = fcmp oeq float %a, %b
  %r = select i1 %cmp, i64 %c, i64 %d
  ret i64 %r
}

; CHECK-LABEL: test_select_fcmp_une_i64:

; CMOV-NEXT:   ucomiss  %xmm1, %xmm0
; CMOV-NEXT:   cmovneq  %rdi, %rsi
; CMOV-NEXT:   cmovpq  %rdi, %rsi
; CMOV-NEXT:   movq  %rsi, %rax
; CMOV-NEXT:   retq

; NOCMOV-NEXT:   flds  8(%esp)
; NOCMOV-NEXT:   flds  4(%esp)
; NOCMOV-NEXT:   fucompp
; NOCMOV-NEXT:   fnstsw  %ax
; NOCMOV-NEXT:   sahf
; NOCMOV-NEXT:   leal  12(%esp), %ecx
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  20(%esp), %ecx
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   movl  (%ecx), %eax
; NOCMOV-NEXT:   movl  4(%ecx), %edx
; NOCMOV-NEXT:   retl
define i64 @test_select_fcmp_une_i64(float %a, float %b, i64 %c, i64 %d) #0 {
entry:
  %cmp = fcmp une float %a, %b
  %r = select i1 %cmp, i64 %c, i64 %d
  ret i64 %r
}

; CHECK-LABEL: test_select_fcmp_oeq_f64:

; CMOV-NEXT:   ucomiss  %xmm1, %xmm0
; CMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; CMOV-NEXT:   jp  [[TBB]]
; CMOV-NEXT:   movaps  %xmm2, %xmm3
; CMOV-NEXT: [[TBB]]:
; CMOV-NEXT:   movaps  %xmm3, %xmm0
; CMOV-NEXT:   retq

; NOCMOV-NEXT:   flds  8(%esp)
; NOCMOV-NEXT:   flds  4(%esp)
; NOCMOV-NEXT:   fucompp
; NOCMOV-NEXT:   fnstsw  %ax
; NOCMOV-NEXT:   sahf
; NOCMOV-NEXT:   leal  20(%esp), %eax
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  12(%esp), %eax
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   fldl  (%eax)
; NOCMOV-NEXT:   retl
define double @test_select_fcmp_oeq_f64(float %a, float %b, double %c, double %d) #0 {
entry:
  %cmp = fcmp oeq float %a, %b
  %r = select i1 %cmp, double %c, double %d
  ret double %r
}

; CHECK-LABEL: test_select_fcmp_oeq_v4i32:

; CMOV-NEXT:   ucomiss  %xmm1, %xmm0
; CMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; CMOV-NEXT:   jp  [[TBB]]
; CMOV-NEXT:   movaps  %xmm2, %xmm3
; CMOV-NEXT: [[TBB]]:
; CMOV-NEXT:   movaps  %xmm3, %xmm0
; CMOV-NEXT:   retq

; NOCMOV-NEXT:   pushl  %edi
; NOCMOV-NEXT:   pushl  %esi
; NOCMOV-NEXT:   flds  20(%esp)
; NOCMOV-NEXT:   flds  16(%esp)
; NOCMOV-NEXT:   fucompp
; NOCMOV-NEXT:   fnstsw  %ax
; NOCMOV-NEXT:   sahf
; NOCMOV-NEXT:   leal  40(%esp), %eax
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  24(%esp), %eax
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   movl  (%eax), %ecx
; NOCMOV-NEXT:   leal  44(%esp), %edx
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  28(%esp), %edx
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   movl  12(%esp), %eax
; NOCMOV-NEXT:   movl  (%edx), %edx
; NOCMOV-NEXT:   leal  48(%esp), %esi
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  32(%esp), %esi
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   movl  (%esi), %esi
; NOCMOV-NEXT:   leal  52(%esp), %edi
; NOCMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; NOCMOV-NEXT:   jp  [[TBB]]
; NOCMOV-NEXT:   leal  36(%esp), %edi
; NOCMOV-NEXT: [[TBB]]:
; NOCMOV-NEXT:   movl  (%edi), %edi
; NOCMOV-NEXT:   movl  %edi, 12(%eax)
; NOCMOV-NEXT:   movl  %esi, 8(%eax)
; NOCMOV-NEXT:   movl  %edx, 4(%eax)
; NOCMOV-NEXT:   movl  %ecx, (%eax)
; NOCMOV-NEXT:   popl  %esi
; NOCMOV-NEXT:   popl  %edi
; NOCMOV-NEXT:   retl  $4
define <4 x i32> @test_select_fcmp_oeq_v4i32(float %a, float %b, <4 x i32> %c, <4 x i32> %d) #0 {
entry:
  %cmp = fcmp oeq float %a, %b
  %r = select i1 %cmp, <4 x i32> %c, <4 x i32> %d
  ret <4 x i32> %r
}

; Also make sure we catch the original code-sequence of interest:

; CMOV: [[ONE_F32_LCPI:.LCPI.*]]:
; CMOV-NEXT:   .long  1065353216

; CHECK-LABEL: test_zext_fcmp_une:
; CMOV-NEXT:   ucomiss  %xmm1, %xmm0
; CMOV-NEXT:   movss  [[ONE_F32_LCPI]](%rip), %xmm0
; CMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; CMOV-NEXT:   jp  [[TBB]]
; CMOV-NEXT:   xorps  %xmm0, %xmm0
; CMOV-NEXT: [[TBB]]:
; CMOV-NEXT:   retq

; NOCMOV:        jne
; NOCMOV-NEXT:   jp
define float @test_zext_fcmp_une(float %a, float %b) #0 {
entry:
  %cmp = fcmp une float %a, %b
  %conv1 = zext i1 %cmp to i32
  %conv2 = sitofp i32 %conv1 to float
  ret float %conv2
}

; CMOV: [[ONE_F32_LCPI:.LCPI.*]]:
; CMOV-NEXT:   .long  1065353216

; CHECK-LABEL: test_zext_fcmp_oeq:
; CMOV-NEXT:   ucomiss  %xmm1, %xmm0
; CMOV-NEXT:   xorps  %xmm0, %xmm0
; CMOV-NEXT:   jne  [[TBB:.LBB[0-9_]+]]
; CMOV-NEXT:   jp  [[TBB]]
; CMOV-NEXT:   movss  [[ONE_F32_LCPI]](%rip), %xmm0
; CMOV-NEXT: [[TBB]]:
; CMOV-NEXT:   retq

; NOCMOV:        jne
; NOCMOV-NEXT:   jp
define float @test_zext_fcmp_oeq(float %a, float %b) #0 {
entry:
  %cmp = fcmp oeq float %a, %b
  %conv1 = zext i1 %cmp to i32
  %conv2 = sitofp i32 %conv1 to float
  ret float %conv2
}

attributes #0 = { nounwind }

@g8 = global i8 0

; The following test failed because llvm had a bug where a structure like:
;
; %12<def> = CMOV_GR8 %7, %11 ... (lt)
; %13<def> = CMOV_GR8 %12, %11 ... (gt)
;
; was lowered to:
;
; The first two cmovs got expanded to:
; BB#0:
;   JL_1 BB#9
; BB#7:
;   JG_1 BB#9
; BB#8:
; BB#9:
;   %12 = phi(%7, BB#8, %11, BB#0, %12, BB#7)
;   %13 = COPY %12
; Which was invalid as %12 is not the same value as %13

; CHECK-LABEL: no_cascade_opt:
; CMOV-DAG: cmpl %edx, %esi
; CMOV-DAG: movb $20, %al
; CMOV-DAG: movb $20, %dl
; CMOV:   jge [[BB2:.LBB[0-9_]+]]
; CMOV:   jle [[BB3:.LBB[0-9_]+]]
; CMOV: [[BB0:.LBB[0-9_]+]]
; CMOV:   testl %edi, %edi
; CMOV:   jne [[BB4:.LBB[0-9_]+]]
; CMOV: [[BB1:.LBB[0-9_]+]]
; CMOV:   movb %al, g8(%rip)
; CMOV:   retq
; CMOV: [[BB2]]:
; CMOV:   movl %ecx, %edx
; CMOV:   jg [[BB0]]
; CMOV: [[BB3]]:
; CMOV:   movl %edx, %eax
; CMOV:   testl %edi, %edi
; CMOV:   je [[BB1]]
; CMOV: [[BB4]]:
; CMOV:   movl %edx, %eax
; CMOV:   movb %al, g8(%rip)
; CMOV:   retq
define void @no_cascade_opt(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
entry:
  %c0 = icmp eq i32 %v0, 0
  %c1 = icmp slt i32 %v1, %v2
  %c2 = icmp sgt i32 %v1, %v2
  %trunc = trunc i32 %v3 to i8
  %sel0 = select i1 %c1, i8 20, i8 %trunc
  %sel1 = select i1 %c2, i8 20, i8 %sel0
  %sel2 = select i1 %c0, i8 %sel1, i8 %sel0
  store volatile i8 %sel2, i8* @g8
  ret void
}
