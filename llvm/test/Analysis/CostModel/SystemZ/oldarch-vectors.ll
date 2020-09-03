; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z10
;
; Check that some costs can be returned for vector instructions also without
; vector support.

define void @fun(<2 x double>* %arg) {
entry:
   %add = fadd <2 x double> undef, undef
   shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
   %conv = fptoui <4 x float> undef to <4 x i32>
   %cmp = icmp eq <2 x i64> undef, undef
  ret void
}
