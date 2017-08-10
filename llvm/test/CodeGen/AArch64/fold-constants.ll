; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

define i64 @dotests_616() {
; CHECK-LABEL: dotests_616
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  fmov x0, d0
; CHECK-NEXT:  ret
entry:
  %0 = bitcast <2 x i64> zeroinitializer to <8 x i16>
  %1 = and <8 x i16> zeroinitializer, %0
  %2 = icmp ne <8 x i16> %1, zeroinitializer
  %3 = extractelement <8 x i1> %2, i32 2
  %vgetq_lane285 = sext i1 %3 to i16
  %vset_lane = insertelement <4 x i16> undef, i16 %vgetq_lane285, i32 0
  %4 = bitcast <4 x i16> %vset_lane to <1 x i64>
  %vget_lane = extractelement <1 x i64> %4, i32 0
  ret i64 %vget_lane
}

; PR25763 - folding constant vector comparisons with sign-extended result
define <8 x i16> @dotests_458() {
; CHECK-LABEL: .LCPI1_0:
; CHECK:       .hword  0                       // 0x0
; CHECK-NEXT:  .hword  0                       // 0x0
; CHECK-NEXT:  .hword  65535                   // 0xffff
; CHECK-NEXT:  .hword  0                       // 0x0
; CHECK-NEXT:  .hword  0                       // 0x0
; CHECK-NEXT:  .hword  0                       // 0x0
; CHECK-NEXT:  .hword  0                       // 0x0
; CHECK-NEXT:  .hword  0                       // 0x0

; CHECK-LABEL: dotests_458
; CHECK:       adrp    x8, .LCPI1_0
; CHECK-NEXT:  ldr     q0, [x8, :lo12:.LCPI1_0]
; CHECK-NEXT:  ret
entry:
  %vclz_v.i = call <8 x i8> @llvm.ctlz.v8i8(<8 x i8> <i8 127, i8 38, i8 -1, i8 -128, i8 127, i8 0, i8 0, i8 0>, i1 false) #6
  %vsra_n = lshr <8 x i8> %vclz_v.i, <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  %name_6 = or <8 x i8> %vsra_n, <i8 127, i8 -128, i8 -1, i8 67, i8 84, i8 127, i8 -1, i8 0>
  %cmp.i603 = icmp slt <8 x i8> %name_6, <i8 -57, i8 -128, i8 127, i8 -128, i8 -1, i8 0, i8 -1, i8 -1>
  %vmovl.i4.i = sext <8 x i1> %cmp.i603 to <8 x i16>
  ret <8 x i16> %vmovl.i4.i
}
declare <8 x i8> @llvm.ctlz.v8i8(<8 x i8>, i1)
