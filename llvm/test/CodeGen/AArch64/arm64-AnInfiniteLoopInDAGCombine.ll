; RUN: llc < %s -mtriple=arm64-eabi

; This test case tests an infinite loop bug in DAG combiner.
; It just tries to do the following replacing endlessly:
; (1)  Replacing.3 0x2c509f0: v4i32 = any_extend 0x2c4cd08 [ORD=4]
;      With: 0x2c4d128: v4i32 = sign_extend 0x2c4cd08 [ORD=4]
;
; (2)  Replacing.2 0x2c4d128: v4i32 = sign_extend 0x2c4cd08 [ORD=4]
;      With: 0x2c509f0: v4i32 = any_extend 0x2c4cd08 [ORD=4]
; As we think the (2) optimization from SIGN_EXTEND to ANY_EXTEND is
; an optimization to replace unused bits with undefined bits, we remove
; the (1) optimization (It doesn't make sense to replace undefined bits
; with signed bits).

define <4 x i32> @infiniteLoop(<4 x i32> %in0, <4 x i16> %in1) {
entry:
  %cmp.i = icmp sge <4 x i16> %in1, <i16 32767, i16 32767, i16 -1, i16 -32768>
  %sext.i = sext <4 x i1> %cmp.i to <4 x i32>
  %mul.i = mul <4 x i32> %in0, %sext.i
  %sext = shl <4 x i32> %mul.i, <i32 16, i32 16, i32 16, i32 16>
  %vmovl.i.i = ashr <4 x i32> %sext, <i32 16, i32 16, i32 16, i32 16>
  ret <4 x i32> %vmovl.i.i
}
