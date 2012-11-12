; RUN: llc < %s > %t
; PR6300
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

; When the "154" loops back onto itself, it defines a register after using it.
; The first value of the register is implicit-def.

%"struct location_chain_def" = type { %"struct location_chain_def"*, %"struct rtx_def"*, %"struct rtx_def"*, i32 }
%"struct real_value" = type { i32, [5 x i32] }
%"struct rtx_def" = type { i16, i8, i8, %"union u" }
%"union u" = type { %"struct real_value" }

define i32 @variable_union(i8** nocapture %slot, i8* nocapture %data) nounwind {
entry:
  br i1 undef, label %"4.thread", label %"3"

"4.thread":                                       ; preds = %entry
  unreachable

"3":                                              ; preds = %entry
  br i1 undef, label %"19", label %"20"

"19":                                             ; preds = %"3"
  unreachable

"20":                                             ; preds = %"3"
  br i1 undef, label %"56.preheader", label %dv_onepart_p.exit

dv_onepart_p.exit:                                ; preds = %"20"
  unreachable

"56.preheader":                                   ; preds = %"20"
  br label %"56"

"50":                                             ; preds = %"57"
  br label %"56"

"56":                                             ; preds = %"50", %"56.preheader"
  br i1 undef, label %"57", label %"58"

"57":                                             ; preds = %"56"
  br i1 undef, label %"50", label %"58"

"58":                                             ; preds = %"57", %"56"
  br i1 undef, label %"62", label %"63"

"62":                                             ; preds = %"58"
  unreachable

"63":                                             ; preds = %"58"
  br i1 undef, label %"67", label %"66"

"66":                                             ; preds = %"63"
  br label %"67"

"67":                                             ; preds = %"66", %"63"
  br label %"68"

"68":                                             ; preds = %"161", %"67"
  br i1 undef, label %"153", label %"161"

"153":                                            ; preds = %"68"
  br i1 undef, label %"160", label %bb.nph46

bb.nph46:                                         ; preds = %"153"
  br label %"154"

"154":                                            ; preds = %"154", %bb.nph46
  %0 = phi %"struct location_chain_def"** [ undef, %bb.nph46 ], [ %1, %"154" ] ; <%"struct location_chain_def"**> [#uses=1]
  %1 = bitcast i8* undef to %"struct location_chain_def"** ; <%"struct location_chain_def"**> [#uses=1]
  store %"struct location_chain_def"* undef, %"struct location_chain_def"** %0, align 4
  br i1 undef, label %"160", label %"154"

"160":                                            ; preds = %"154", %"153"
  br label %"161"

"161":                                            ; preds = %"160", %"68"
  br label %"68"
}
