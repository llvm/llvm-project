; RUN: llc < %s -x86-use-vzeroupper -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

declare <4 x float> @do_sse(<4 x float>)
declare <8 x float> @do_avx(<8 x float>)
declare <4 x float> @llvm.x86.avx.vextractf128.ps.256(<8 x float>, i8) nounwind readnone
@x = common global <4 x float> zeroinitializer, align 16
@g = common global <8 x float> zeroinitializer, align 32

;; Basic checking - don't emit any vzeroupper instruction

; CHECK: _test00
define <4 x float> @test00(<4 x float> %a, <4 x float> %b) nounwind uwtable ssp {
entry:
  ; CHECK-NOT: vzeroupper
  %add.i = fadd <4 x float> %a, %b
  %call3 = call <4 x float> @do_sse(<4 x float> %add.i) nounwind
  ; CHECK: ret
  ret <4 x float> %call3
}

;; Check parameter 256-bit parameter passing

; CHECK: _test01
define <8 x float> @test01(<4 x float> %a, <4 x float> %b, <8 x float> %c) nounwind uwtable ssp {
entry:
  %tmp = load <4 x float>* @x, align 16
  ; CHECK: vzeroupper
  ; CHECK-NEXT: callq _do_sse
  %call = tail call <4 x float> @do_sse(<4 x float> %tmp) nounwind
  store <4 x float> %call, <4 x float>* @x, align 16
  ; CHECK-NOT: vzeroupper
  ; CHECK: callq _do_sse
  %call2 = tail call <4 x float> @do_sse(<4 x float> %call) nounwind
  store <4 x float> %call2, <4 x float>* @x, align 16
  ; CHECK: ret
  ret <8 x float> %c
}

;; Test the pass convergence and also that vzeroupper is only issued when necessary,
;; for this function it should be only once

; CHECK: _test02
define <4 x float> @test02(<4 x float> %a, <4 x float> %b) nounwind uwtable ssp {
entry:
  %add.i = fadd <4 x float> %a, %b
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  ; CHECK: LBB
  ; CHECK-NOT: vzeroupper
  %i.018 = phi i32 [ 0, %entry ], [ %1, %for.body ]
  %c.017 = phi <4 x float> [ %add.i, %entry ], [ %call14, %for.body ]
  ; CHECK: callq _do_sse
  %call5 = tail call <4 x float> @do_sse(<4 x float> %c.017) nounwind
  ; CHECK-NEXT: callq _do_sse
  %call7 = tail call <4 x float> @do_sse(<4 x float> %call5) nounwind
  %tmp11 = load <8 x float>* @g, align 32
  %0 = tail call <4 x float> @llvm.x86.avx.vextractf128.ps.256(<8 x float> %tmp11, i8 1) nounwind
  ; CHECK: vzeroupper
  ; CHECK-NEXT: callq _do_sse
  %call14 = tail call <4 x float> @do_sse(<4 x float> %0) nounwind
  %1 = add nsw i32 %i.018, 1
  %exitcond = icmp eq i32 %1, 4
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret <4 x float> %call14
}

;; Check that we also perform vzeroupper when we return from a function.

; CHECK: _test03
define <4 x float> @test03(<4 x float> %a, <4 x float> %b) nounwind uwtable ssp {
entry:
  %shuf = shufflevector <4 x float> %a, <4 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; CHECK-NOT: vzeroupper
  ; CHECK: call
  %call = call <8 x float> @do_avx(<8 x float> %shuf) nounwind
  %shuf2 = shufflevector <8 x float> %call, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: vzeroupper
  ; CHECK: ret
  ret <4 x float> %shuf2
}
