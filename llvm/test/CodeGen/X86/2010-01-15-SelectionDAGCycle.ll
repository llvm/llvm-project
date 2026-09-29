; RUN: llc < %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @numvec_(ptr noalias %ncelet, ptr noalias %ncel, ptr noalias %nfac, ptr noalias %nfabor, ptr noalias %lregis, ptr noalias %irveci, ptr noalias %irvecb, ptr noalias %ifacel, ptr noalias %ifabor, ptr noalias %inumfi, ptr noalias %inumfb, ptr noalias %iworkf, ptr noalias %ismbs) {
"file bug754399.f90, line 1, bb1":
	%r1037 = bitcast <2 x double> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=1]
	br label %"file bug754399.f90, line 184, in inner vector loop at depth 0, bb164"

"file bug754399.f90, line 184, in inner vector loop at depth 0, bb164":		; preds = %"file bug754399.f90, line 184, in inner vector loop at depth 0, bb164", %"file bug754399.f90, line 1, bb1"
	%tmp641 = add i64 0, 48		; <i64> [#uses=1]
	%tmp641642 = inttoptr i64 %tmp641 to ptr		; <ptr> [#uses=1]
	%r1258 = load <4 x i32>, ptr %tmp641642, align 4		; <<4 x i32>> [#uses=2]
	%r1295 = extractelement <4 x i32> %r1258, i32 3		; <i32> [#uses=1]
	%r1296 = sext i32 %r1295 to i64		; <i64> [#uses=1]
	%r1297 = add i64 %r1296, -1		; <i64> [#uses=1]
	%r1298183 = getelementptr [0 x i32], ptr %ismbs, i64 0, i64 %r1297		; <ptr> [#uses=1]
	%r1298184 = load i32, ptr %r1298183, align 4		; <i32> [#uses=1]
	%r1301 = extractelement <4 x i32> %r1037, i32 3		; <i32> [#uses=1]
	%r1302 = mul i32 %r1298184, %r1301		; <i32> [#uses=1]
	%r1306 = insertelement <4 x i32> zeroinitializer, i32 %r1302, i32 3		; <<4 x i32>> [#uses=1]
	%r1321 = add <4 x i32> %r1306, %r1258		; <<4 x i32>> [#uses=1]
	%tmp643 = add i64 0, 48		; <i64> [#uses=1]
	%tmp643644 = inttoptr i64 %tmp643 to ptr		; <ptr> [#uses=1]
	store <4 x i32> %r1321, ptr %tmp643644, align 4
	br label %"file bug754399.f90, line 184, in inner vector loop at depth 0, bb164"
}
