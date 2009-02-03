; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; PR2256
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-mingw32"

define { x86_fp80, x86_fp80 } @catanl({ x86_fp80, x86_fp80 }* byval  %Z, i1 %cond) nounwind  {
bb:		; preds = %entry
	br i1  %cond, label %bb48, label %bb40

bb40:		; preds = %bb
	store i32 34, i32* null, align 4
	br label %bb196

bb48:		; preds = %bb.bb48_crit_edge, %entry.bb48_crit_edge
	%tmp53 = icmp eq i32 0, 1280		; <i1> [#uses=1]
	br i1 %tmp53, label %bb56, label %bb174

bb56:		; preds = %bb48
	%iftmp.0.0 = select i1 false, x86_fp80 0xK3FFFC90FDAA22168C235, x86_fp80 0xKBFFFC90FDAA22168C235		; <x86_fp80> [#uses=0]
	br label %bb196


bb174:		; preds = %bb144, %bb114
	%tmp191 = mul x86_fp80 0xK00000000000000000000, 0xK3FFE8000000000000000		; <x86_fp80> [#uses=1]
	br label %bb196

bb196:		; preds = %bb174, %bb56, %bb40
	%Res.1.0 = phi x86_fp80 [ 0xK7FFF8000000000000000, %bb40 ], [ %tmp191, %bb174 ], [ 0xK00000000000000000000, %bb56 ]		; <x86_fp80> [#uses=1]
	ret x86_fp80 0xK00000000000000000000, x86_fp80 %Res.1.0
}
