; RUN: llc < %s -march=x86-64 -mattr=+sse42 |  FileCheck %s


define float @extractFloat1() nounwind {
entry:
  ; CHECK: 1065353216
  %tmp0 = bitcast <1 x double> <double 0x000000003F800000> to <2 x float>
  %tmp1 = extractelement <2 x float> %tmp0, i32 0 
  ret float %tmp1
}

define float @extractFloat2() nounwind {
entry:
  ; CHECK: xorps	%xmm0, %xmm0
  %tmp4 = bitcast <1 x double> <double 0x000000003F800000> to <2 x float>
  %tmp5 = extractelement <2 x float> %tmp4, i32 1
  ret float %tmp5
}

define i32 @extractInt2() nounwind {
entry:
  ; CHECK: xorl	%eax, %eax
  %tmp4 = bitcast <1 x i64> <i64 256> to <2 x i32>
  %tmp5 = extractelement <2 x i32> %tmp4, i32 1
  ret i32 %tmp5
}

