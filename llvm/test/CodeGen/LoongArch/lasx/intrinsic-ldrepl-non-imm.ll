; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(ptr, i32)

define <32 x i8> @lasx_xvldrepl_b(ptr %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(ptr %p, i32 %a)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(ptr, i32)

define <16 x i16> @lasx_xvldrepl_h(ptr %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(ptr %p, i32 %a)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(ptr, i32)

define <8 x i32> @lasx_xvldrepl_w(ptr %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(ptr %p, i32 %a)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(ptr, i32)

define <4 x i64> @lasx_xvldrepl_d(ptr %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(ptr %p, i32 %a)
  ret <4 x i64> %res
}
