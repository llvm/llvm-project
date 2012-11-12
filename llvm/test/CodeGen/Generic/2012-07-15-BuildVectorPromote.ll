; RUN: llc -mcpu=corei7 < %s
; We don't care about the output, just that it doesn't crash

define <1 x i1> @buildvec_promote() {
  %cmp = icmp ule <1 x i32> undef, undef
  %sel = select i1 undef, <1 x i1> undef, <1 x i1> %cmp
  ret <1 x i1> %sel
}
