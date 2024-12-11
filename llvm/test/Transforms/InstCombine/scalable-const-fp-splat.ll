; RUN: opt -passes=instcombine -S -o - < %s | FileCheck %s

define <vscale x 2 x float> @shrink_splat_scalable_extend(<vscale x 2 x float> %a) {
  ; CHECK-LABEL: @shrink_splat_scalable_extend
  ; CHECK-NEXT:  %[[FADD:.*]] = fadd <vscale x 2 x float> %a, shufflevector (<vscale x 2 x float> insertelement (<vscale x 2 x float> undef, float -1.000000e+00, i32 0), <vscale x 2 x float> undef, <vscale x 2 x i32> zeroinitializer)
  ; CHECK-NEXT:  ret <vscale x 2 x float> %[[FADD]]
  %1 = shufflevector <vscale x 2 x float> insertelement (<vscale x 2 x float> undef, float -1.000000e+00, i32 0), <vscale x 2 x float> undef, <vscale x 2 x i32> zeroinitializer
  %2 = fpext <vscale x 2 x float> %a to <vscale x 2 x double>
  %3 = fpext <vscale x 2 x float> %1 to <vscale x 2 x double>
  %4 = fadd <vscale x 2 x double> %2, %3
  %5 = fptrunc <vscale x 2 x double> %4 to <vscale x 2 x float>
  ret <vscale x 2 x float> %5
}
