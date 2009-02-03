; RUN: llvm-as < %s | llc -march=x86-64

declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>)
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)

define <2 x i64> @footz(<2 x i64> %a) nounwind {
  %c = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a)
  ret <2 x i64> %c
}
define <2 x i64> @foolz(<2 x i64> %a) nounwind {
  %c = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a)
  ret <2 x i64> %c
}
define <2 x i64> @foopop(<2 x i64> %a) nounwind {
  %c = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)
  ret <2 x i64> %c
}
