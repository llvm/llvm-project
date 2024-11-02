; RUN: llc < %s -mtriple=i686-unknown -mattr=+avx
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx

;
; llvm-stress generated crash case due to build_vector implicit
; truncation bug from constant folding after legalization.
;

@G = external dso_local global i32

define void @bv_crash_test() {
  %I = insertelement <4 x i64> zeroinitializer, i64 15910, i32 0
  %Tr = trunc <4 x i64> %I to <4 x i8>
  %Bc = bitcast <4 x i8> %Tr to i32
  store volatile i32 %Bc, ptr @G
  ret void
}
