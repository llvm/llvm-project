; RUN: llc < %s -march=x86
; PR3453

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%struct.cl_engine = type { i32, i16, i32, i8**, i8**, i8*, i8*, i8*, i8*, i8*, i8*, i8* }
	%struct.cl_limits = type { i32, i32, i32, i32, i16, i32 }
	%struct.cli_ac_alt = type { i8, i8*, i16, i16, %struct.cli_ac_alt* }
	%struct.cli_ac_node = type { i8, i8, %struct.cli_ac_patt*, %struct.cli_ac_node**, %struct.cli_ac_node* }
	%struct.cli_ac_patt = type { i16*, i16*, i16, i16, i8, i32, i32, i8*, i8*, i32, i16, i16, i16, i16, %struct.cli_ac_alt**, i8, i16, %struct.cli_ac_patt*, %struct.cli_ac_patt* }
	%struct.cli_bm_patt = type { i8*, i8*, i16, i16, i8*, i8*, i8, %struct.cli_bm_patt*, i16 }
	%struct.cli_ctx = type { i8**, i32*, %struct.cli_matcher*, %struct.cl_engine*, %struct.cl_limits*, i32, i32, i32, i32, %struct.cli_dconf* }
	%struct.cli_dconf = type { i32, i32, i32, i32, i32, i32, i32 }
	%struct.cli_matcher = type { i16, i8, i8*, %struct.cli_bm_patt**, i32*, i32, i8, i8, %struct.cli_ac_node*, %struct.cli_ac_node**, %struct.cli_ac_patt**, i32, i32, i32 }

define fastcc i32 @cli_scanautoit(i32 %desc, %struct.cli_ctx* %ctx, i32 %offset) nounwind {
entry:
	br i1 false, label %bb.i49.i72, label %bb14

bb.i49.i72:		; preds = %bb.i49.i72, %entry
	%UNP.i1482.0 = phi i288 [ %.ins659, %bb.i49.i72 ], [ undef, %entry ]		; <i288> [#uses=1]
	%0 = load i32* null, align 4		; <i32> [#uses=1]
	%1 = xor i32 %0, 17834		; <i32> [#uses=1]
	%2 = zext i32 %1 to i288		; <i288> [#uses=1]
	%3 = shl i288 %2, 160		; <i288> [#uses=1]
	%UNP.i1482.in658.mask = and i288 %UNP.i1482.0, -6277101733925179126504886505003981583386072424808101969921		; <i288> [#uses=1]
	%.ins659 = or i288 %3, %UNP.i1482.in658.mask		; <i288> [#uses=1]
	br label %bb.i49.i72

bb14:		; preds = %entry
	ret i32 -123
}
