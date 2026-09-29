; RUN: llc < %s -mtriple=i686--
; PR3243

declare signext i16 @safe_mul_func_int16_t_s_s(i16 signext, i32) nounwind readnone optsize

define i32 @func_120(i32 %p_121) nounwind optsize {
entry:
	%0 = trunc i32 %p_121 to i16		; <i16> [#uses=1]
	%1 = urem i16 %0, -15461		; <i16> [#uses=1]
	%phitmp1 = trunc i16 %1 to i8		; <i8> [#uses=1]
	%phitmp2 = urem i8 %phitmp1, -1		; <i8> [#uses=1]
	%phitmp3 = zext i8 %phitmp2 to i16		; <i16> [#uses=1]
	%2 = tail call signext i16 @safe_mul_func_int16_t_s_s(i16 signext %phitmp3, i32 1) nounwind		; <i16> [#uses=0]
	unreachable
}
