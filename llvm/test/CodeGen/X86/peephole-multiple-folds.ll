; RUN: llc -mtriple=x86_64-- -mcpu=core-avx2 < %s | FileCheck %s
;
; Test multiple peephole-time folds in a single basic block.
; <rdar://problem/16478629>

define <8 x float> @test_peephole_multi_fold(<8 x float>* %p1, <8 x float>* %p2) {
entry:
  br label %loopbody

loopbody:
; CHECK: test_peephole_multi_fold:
; CHECK: vfmadd231ps ({{%rdi|%rcx}}),
; CHECK: vfmadd231ps ({{%rsi|%rdx}}),
  %vsum1 = phi <8 x float> [ %vsum1.next, %loopbody ], [ zeroinitializer, %entry ]
  %vsum2 = phi <8 x float> [ %vsum2.next, %loopbody ], [ zeroinitializer, %entry ]
  %m1 = load <8 x float>, <8 x float>* %p1, align 1
  %m2 = load <8 x float>, <8 x float>* %p2, align 1
  %vsum1.next = tail call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %m1, <8 x float> zeroinitializer, <8 x float> %vsum1)
  %vsum2.next = tail call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %m2, <8 x float> zeroinitializer, <8 x float> %vsum2)
  %vsum1.next.1 = extractelement <8 x float> %vsum1.next, i32 0
  %c = fcmp oeq float %vsum1.next.1, 0.0
  br i1 %c, label %loopbody, label %loopexit

loopexit:
  %r = fadd <8 x float> %vsum1.next, %vsum2.next
  ret <8 x float> %r
}

declare <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>)
