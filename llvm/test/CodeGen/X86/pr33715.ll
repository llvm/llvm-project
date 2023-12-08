; Make sure we don't crash with a build vector of integer constants.
; RUN: llc %s -o /dev/null

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @patatino() {
  %tmp = insertelement <4 x i32> <i32 1, i32 1, i32 undef, i32 undef>, i32 1, i32 2
  %tmp1 = insertelement <4 x i32> %tmp, i32 1, i32 3
  %tmp2 = icmp ne <4 x i32> %tmp1, zeroinitializer
  %tmp3 = icmp slt <4 x i32> %tmp1, <i32 4, i32 4, i32 4, i32 4>
  %tmp4 = or <4 x i1> %tmp2, %tmp3
  %tmp5 = select <4 x i1> %tmp4, <4 x i32> zeroinitializer, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %tmp6 = extractelement <4 x i32> %tmp5, i32 0
  ret i32 %tmp6
}
