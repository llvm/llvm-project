; RUN: llc < %s
; rdar://6505632
; reduced from 483.xalancbmk

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%"struct.std::basic_ostream<char,std::char_traits<char> >.base" = type { i32 (...)** }
	%"struct.xercesc_2_5::ASCIIRangeFactory" = type { %"struct.std::basic_ostream<char,std::char_traits<char> >.base", i8, i8 }
@_ZN11xercesc_2_5L17gIdeographicCharsE = external constant [7 x i16]		; <[7 x i16]*> [#uses=3]

define void @_ZN11xercesc_2_515XMLRangeFactory11buildRangesEv(%"struct.xercesc_2_5::ASCIIRangeFactory"* %this) {
entry:
	br i1 false, label %bb5, label %return

bb5:		; preds = %entry
	br label %bb4.i.i

bb4.i.i:		; preds = %bb4.i.i, %bb5
	br i1 false, label %bb.i51, label %bb4.i.i

bb.i51:		; preds = %bb.i51, %bb4.i.i
	br i1 false, label %bb4.i.i70, label %bb.i51

bb4.i.i70:		; preds = %bb4.i.i70, %bb.i51
	br i1 false, label %_ZN11xercesc_2_59XMLString9stringLenEPKt.exit.i73, label %bb4.i.i70

_ZN11xercesc_2_59XMLString9stringLenEPKt.exit.i73:		; preds = %bb4.i.i70
	%0 = load i16* getelementptr ([7 x i16]* @_ZN11xercesc_2_5L17gIdeographicCharsE, i32 0, i32 add (i32 ashr (i32 sub (i32 ptrtoint (i16* getelementptr ([7 x i16]* @_ZN11xercesc_2_5L17gIdeographicCharsE, i32 0, i32 4) to i32), i32 ptrtoint ([7 x i16]* @_ZN11xercesc_2_5L17gIdeographicCharsE to i32)), i32 1), i32 1)), align 4		; <i16> [#uses=0]
	br label %bb4.i5.i141

bb4.i5.i141:		; preds = %bb4.i5.i141, %_ZN11xercesc_2_59XMLString9stringLenEPKt.exit.i73
	br label %bb4.i5.i141

return:		; preds = %entry
	ret void
}
