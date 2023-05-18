; RUN: llc %s -O0 -fast-isel -regalloc=fast -mcpu=i386 -o -
; PR4767

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10"

define void @fn(x86_fp80 %x) nounwind ssp {
entry:
  %x.addr = alloca x86_fp80                       ; <ptr> [#uses=5]
  store x86_fp80 %x, ptr %x.addr
  br i1 false, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %tmp = load x86_fp80, ptr %x.addr                   ; <x86_fp80> [#uses=1]
  %tmp1 = load x86_fp80, ptr %x.addr                  ; <x86_fp80> [#uses=1]
  %cmp = fcmp oeq x86_fp80 %tmp, %tmp1            ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %if.end

cond.false:                                       ; preds = %entry
  %tmp2 = load x86_fp80, ptr %x.addr                  ; <x86_fp80> [#uses=1]
  %tmp3 = load x86_fp80, ptr %x.addr                  ; <x86_fp80> [#uses=1]
  %cmp4 = fcmp une x86_fp80 %tmp2, %tmp3          ; <i1> [#uses=1]
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %cond.false, %cond.true
  br label %if.end

if.end:                                           ; preds = %if.then, %cond.false, %cond.true
  ret void
}

; PR10575
; This produces a FP0 = IMPLICIT_DEF instruction.
define void @__m_rankmerge_MOD_dindexmerge_() nounwind {
entry:
  br label %"20"

"20":                                             ; preds = %"23", %entry
    %0 = phi double [ undef, %entry ], [ %0, %"23" ]
    %1 = phi double [ 0.000000e+00, %entry ], [ %2, %"23" ]
    br i1 undef, label %"21", label %"23"

"21":                                             ; preds = %"20"
    ret void

"23":                                             ; preds = %"20"
    %2 = select i1 undef, double %0, double %1
    br label %"20"
}
