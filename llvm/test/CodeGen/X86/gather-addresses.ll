; RUN: llc -mtriple=x86_64-linux -mcpu=nehalem < %s | FileCheck %s --check-prefix=LIN
; RUN: llc -mtriple=x86_64-win32 -mcpu=nehalem < %s | FileCheck %s --check-prefix=WIN
; RUN: llc -mtriple=i686-win32 -mcpu=nehalem < %s | FileCheck %s --check-prefix=LIN32
; rdar://7398554

; When doing vector gather-scatter index calculation with 32-bit indices,
; use an efficient mov/shift sequence rather than shuffling each individual
; element out of the index vector.

; CHECK-LABEL: foo:
; LIN: movdqa	(%rsi), %xmm0
; LIN: pand 	(%rdx), %xmm0
; LIN: pextrq	$1, %xmm0, %r[[REG4:.+]]
; LIN: movq 	%xmm0, %r[[REG2:.+]]
; LIN: movslq	%e[[REG2]], %r[[REG1:.+]]
; LIN: sarq    $32, %r[[REG2]]
; LIN: movslq	%e[[REG4]], %r[[REG3:.+]]
; LIN: sarq    $32, %r[[REG4]]
; LIN: movsd    (%rdi,%r[[REG3]],8), %xmm1
; LIN: movhpd   (%rdi,%r[[REG4]],8), %xmm1 
; LIN: movq     %rdi, %xmm1 
; LIN: movq     %r[[REG3]], %xmm0

; WIN: movdqa	(%rdx), %xmm0
; WIN: pand 	(%r8), %xmm0
; WIN: pextrq	$1, %xmm0, %r[[REG4:.+]]
; WIN: movq 	%xmm0, %r[[REG2:.+]]
; WIN: movslq	%e[[REG2]], %r[[REG1:.+]]
; WIN: sarq    $32, %r[[REG2]]
; WIN: movslq	%e[[REG4]], %r[[REG3:.+]]
; WIN: sarq    $32, %r[[REG4]]
; WIN: movsd    (%rcx,%r[[REG3]],8), %xmm1
; WIN: movhpd   (%rcx,%r[[REG4]],8), %xmm1
; WIN: movdqa   (%r[[REG2]]), %xmm0
; WIN: movq     %r[[REG2]], %xmm1

define <4 x double> @foo(double* %p, <4 x i32>* %i, <4 x i32>* %h) nounwind {
  %a = load <4 x i32>, <4 x i32>* %i
  %b = load <4 x i32>, <4 x i32>* %h
  %j = and <4 x i32> %a, %b
  %d0 = extractelement <4 x i32> %j, i32 0
  %d1 = extractelement <4 x i32> %j, i32 1
  %d2 = extractelement <4 x i32> %j, i32 2
  %d3 = extractelement <4 x i32> %j, i32 3
  %q0 = getelementptr double, double* %p, i32 %d0
  %q1 = getelementptr double, double* %p, i32 %d1
  %q2 = getelementptr double, double* %p, i32 %d2
  %q3 = getelementptr double, double* %p, i32 %d3
  %r0 = load double, double* %q0
  %r1 = load double, double* %q1
  %r2 = load double, double* %q2
  %r3 = load double, double* %q3
  %v0 = insertelement <4 x double> undef, double %r0, i32 0
  %v1 = insertelement <4 x double> %v0, double %r1, i32 1
  %v2 = insertelement <4 x double> %v1, double %r2, i32 2
  %v3 = insertelement <4 x double> %v2, double %r3, i32 3
  ret <4 x double> %v3
}

; Check that the sequence previously used above, which bounces the vector off the
; cache works for x86-32. Note that in this case it will not be used for index
; calculation, since indexes are 32-bit, not 64.
; CHECK-LABEL: old:
; LIN32: movaps	%xmm0, (%esp)
; LIN32-DAG: {{(mov|and)}}l	(%esp),
; LIN32-DAG: {{(mov|and)}}l	4(%esp),
; LIN32-DAG: {{(mov|and)}}l	8(%esp),
; LIN32-DAG: {{(mov|and)}}l	12(%esp),
define <4 x i64> @old(double* %p, <4 x i32>* %i, <4 x i32>* %h, i64 %f) nounwind {
  %a = load <4 x i32>, <4 x i32>* %i
  %b = load <4 x i32>, <4 x i32>* %h
  %j = and <4 x i32> %a, %b
  %d0 = extractelement <4 x i32> %j, i32 0
  %d1 = extractelement <4 x i32> %j, i32 1
  %d2 = extractelement <4 x i32> %j, i32 2
  %d3 = extractelement <4 x i32> %j, i32 3
  %q0 = zext i32 %d0 to i64
  %q1 = zext i32 %d1 to i64
  %q2 = zext i32 %d2 to i64
  %q3 = zext i32 %d3 to i64  
  %r0 = and i64 %q0, %f
  %r1 = and i64 %q1, %f
  %r2 = and i64 %q2, %f
  %r3 = and i64 %q3, %f
  %v0 = insertelement <4 x i64> undef, i64 %r0, i32 0
  %v1 = insertelement <4 x i64> %v0, i64 %r1, i32 1
  %v2 = insertelement <4 x i64> %v1, i64 %r2, i32 2
  %v3 = insertelement <4 x i64> %v2, i64 %r3, i32 3
  ret <4 x i64> %v3
}
