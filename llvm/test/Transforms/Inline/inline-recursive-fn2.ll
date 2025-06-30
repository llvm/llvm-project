; REQUIRES: asserts
; RUN: opt -passes='cgscc(inline),instcombine,cgscc(inline)' -S -debug-only=inline -disable-output < %s 2>&1 | FileCheck %s

; This test shows that the recursive function will not get simplified
; unless the caller is the function itself, not another different caller.

; CHECK:  Inlining calls in: test
; CHECK:      Function size: 2
; CHECK:      NOT Inlining (cost=never): recursive, Call:   %call = tail call float @inline_rec_true_successor(float %x, float %scale)

; CHECK:  Inlining calls in: inline_rec_true_successor
; CHECK:      Function size: 10
; CHECK:      Inlining (cost=-35, threshold=337), Call:   %call = tail call float @inline_rec_true_successor(float %fneg, float %scale)
; CHECK:      Size after inlining: 17
; CHECK:      NOT Inlining (cost=never): noinline function attribute, Call:   %call_test = tail call float @test(float %fneg, float %common.ret18.op.i)
; CHECK:      NOT Inlining (cost=never): noinline function attribute, Call:   %call_test.i = tail call float @test(float %x, float %call.i)
; CHECK:  Skipping inlining due to history: inline_rec_true_successor -> inline_rec_true_successor
; CHECK:  Updated inlining SCC: (test, inline_rec_true_successor)

; CHECK:  Inlining calls in: test
; CHECK:      Function size: 2
; CHECK:      Inlining (cost=25, threshold=225), Call:   %call = tail call float @inline_rec_true_successor(float %x, float %scale)
; CHECK:      Size after inlining: 10

define float @test(float %x, float %scale) noinline {
entry:
  %call = tail call float @inline_rec_true_successor(float %x, float %scale)
  ret float %call
}

define float @inline_rec_true_successor(float %x, float %scale)  {
entry:
  %cmp = fcmp olt float %x, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

common.ret18:                                     ; preds = %if.then, %if.end
  %common.ret18.op = phi float [ %call_test, %if.then ], [ %mul, %if.end ]
  ret float %common.ret18.op

if.then:                                          ; preds = %entry
  %fneg = fneg float %x
  %call = tail call float @inline_rec_true_successor(float %fneg, float %scale)
  %call_test = tail call float @test(float %fneg, float %call)
  br label %common.ret18

if.end:                                           ; preds = %entry
  %mul = fmul float %x, %scale
  br label %common.ret18
}
