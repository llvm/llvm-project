; RUN: llc < %s -march=x86 -no-integrated-as

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @dct_unquantize_h263_intra_mmx(i16* %block, i32 %n, i32 %qscale) nounwind  {
entry:
	%tmp1 = shl i32 %qscale, 1		; <i32> [#uses=1]
	br i1 false, label %bb46, label %bb59

bb46:		; preds = %entry
	ret void

bb59:		; preds = %entry
	tail call void asm sideeffect "movd $1, %mm6                 \0A\09packssdw %mm6, %mm6          \0A\09packssdw %mm6, %mm6          \0A\09movd $2, %mm5                 \0A\09pxor %mm7, %mm7              \0A\09packssdw %mm5, %mm5          \0A\09packssdw %mm5, %mm5          \0A\09psubw %mm5, %mm7             \0A\09pxor %mm4, %mm4              \0A\09.align 1<<4\0A\091:                             \0A\09movq ($0, $3), %mm0           \0A\09movq 8($0, $3), %mm1          \0A\09pmullw %mm6, %mm0            \0A\09pmullw %mm6, %mm1            \0A\09movq ($0, $3), %mm2           \0A\09movq 8($0, $3), %mm3          \0A\09pcmpgtw %mm4, %mm2           \0A\09pcmpgtw %mm4, %mm3           \0A\09pxor %mm2, %mm0              \0A\09pxor %mm3, %mm1              \0A\09paddw %mm7, %mm0             \0A\09paddw %mm7, %mm1             \0A\09pxor %mm0, %mm2              \0A\09pxor %mm1, %mm3              \0A\09pcmpeqw %mm7, %mm0           \0A\09pcmpeqw %mm7, %mm1           \0A\09pandn %mm2, %mm0             \0A\09pandn %mm3, %mm1             \0A\09movq %mm0, ($0, $3)           \0A\09movq %mm1, 8($0, $3)          \0A\09add $$16, $3                    \0A\09jng 1b                         \0A\09", "r,imr,imr,r,~{dirflag},~{fpsr},~{flags},~{memory}"( i16* null, i32 %tmp1, i32 0, i32 0 ) nounwind 
	ret void
}
