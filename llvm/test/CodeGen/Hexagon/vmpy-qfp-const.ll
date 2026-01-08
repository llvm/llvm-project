; In this example operands in fmul instruction are (fpext, constant_vector). The generated assembly
; should contains vsplat instruction followed by multiplication of two halfs whose result is of type qf32.
; RUN: llc -march=hexagon -mattr=+hvxv69,+hvx-length128b < %s | FileCheck %s

; CHECK-LABEL: check1
; CHECK: [[REG0:(r[0-9]+)]] = ##
; CHECK: [[VREG0:(v[0-9]+)]] = vsplat([[REG0]])
; CHECK: v{{[0-9:]+}}.qf32 = vmpy(v{{[0-9]+}}.hf,[[VREG0]].hf)

; Function Attrs: norecurse nounwind
define dso_local void @check1(half* nocapture readonly %a, float* nocapture %r) local_unnamed_addr {
entry:
  %0 = bitcast half* %a to <64 x half>*
  %wide.load = load <64 x half>, <64 x half>* %0, align 2
  %1 = fpext <64 x half> %wide.load to <64 x float>
  %2= fmul <64 x float> %1, <float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000, float 0x3FC9980000000000>
  %3 = bitcast float* %r to <64 x float>*
  store <64 x float> %2, <64 x float>* %3, align 4
  ret void
}

; Widening float vector with vector-width 128
; CHECK-LABEL: check2
; CHECK: v{{[0-9:]+}}.qf32 = vmpy(v{{[0-9]+}}.hf,[[VREG1:(v[0-9]+)]].hf)
; CHECK: [[VREG1]].cur = vmem(r{{[0-9\+\#0-9]+}})
; CHECK: v{{[0-9:]+}}.qf32 = vmpy(v{{[0-9]+}}.hf,[[VREG1]].hf)
define dllexport void @check2(i8* noalias nocapture writeonly align 128 %0, i8* noalias nocapture readonly align 128 %1) #0 {
  %3 = bitcast i8* %0 to <128 x float>*
  %4 = bitcast i8* %1 to <128 x half>*
  %5 = load <128 x half>, <128 x half>* %4, align 128
  %6 = fpext <128 x half> %5 to <128 x float>
  %7 = fmul nnan nsz <128 x float> %6, <float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01>
  store <128 x float> %7, <128 x float>* %3, align 128
  ret void
}

; Widening float vector pass do not handle instructions with
; vector-width > 128. Instead during ISel, FPExtend on the operands
; of FMUL will generate sequence of vmpy_qf32_hf, shuffle inst to
; convert float16 to float32.
; Later, vmpy_qf32_sf instruction will be generated for multiplying
; two operands of FMUL instruction.
; CHECK-LABEL: check3
; CHECK: v{{[0-9:]+}}.qf32 = vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)
define dllexport void @check3(i8* noalias nocapture writeonly align 256 %0, i8* noalias nocapture readonly align 256 %1) #0 {
  %3 = bitcast i8* %0 to <256 x float>*
  %4 = bitcast i8* %1 to <256 x half>*
  %5 = load <256 x half>, <256 x half>* %4, align 128
  %6 = fpext <256 x half> %5 to <256 x float>
  %7 = fmul nnan nsz <256 x float> %6, <float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01, float 1.000000e+00, float 7.500000e-01, float 5.000000e-01, float 2.500000e-01>
  store <256 x float> %7, <256 x float>* %3, align 128
  ret void
}

; Widening float vector 32xf16
; check4 also serve as a test case for HexagonOptShuffleVector with single Hi/Lo use case, where the pass should prevent relocating shuffle instruction generated by HexagonGenWideningVecFloatInstr, otherwise the function will be broken.
; CHECK-LABEL: check4
; CHECK: v{{[0-9:]+}}.qf32 = vmpy(v{{[0-9]+}}.hf,v{{[0-9:]+}}.hf)
; CHECK: v{{[0-9:]+}} = vshuff(v{{[0-9]+}},v{{[0-9:]+}},r{{[0-9]+}})
define dso_local void @check4(half* nocapture readonly %a, half* nocapture readonly %b, float* nocapture %r) local_unnamed_addr #0 {
entry:
  %0 = bitcast half* %a to <32 x half>*
  %wide.load.0 = load <32 x half>, <32 x half>* %0, align 2
  %1 = bitcast half* %b to <32 x half>*
  %wide.load.1 = load <32 x half>, <32 x half>* %1, align 2
  %2 = fpext <32 x half> %wide.load.0 to <32 x float>
  %3 = fpext <32 x half> %wide.load.1 to <32 x float>
  %4= fmul <32 x float> %2, %3
  store <32 x float> %4, <32 x float>* %r, align 4
  ret void
}
