; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names < %s | FileCheck %s

 define float @FloatConstantPool() {
; CHECK-LABEL: FloatConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plfs f1, .LCPI0_0@PCREL(0), 1
entry:
  ret float 0x380FFFF840000000
}

 define double @DoubleConstantPool() {
; CHECK-LABEL: DoubleConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plfd f1, .LCPI1_0@PCREL(0), 1
entry:
  ret double 2.225070e-308
}

 define ppc_fp128 @LongDoubleConstantPool() {
; CHECK-LABEL: LongDoubleConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plfd f1, .LCPI2_0@PCREL(0), 1
; CHECK-NEXT:    plfd f2, .LCPI2_1@PCREL(0), 1
entry:
  ret ppc_fp128 0xM03600000DBA876CC800D16974FD9D27B
}

 define fp128 @__Float128ConstantPool() {
; CHECK-LABEL: __Float128ConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI3_0@PCREL(0), 1
entry:
  ret fp128 0xL00000000000000003C00FFFFC5D02B3A
}

 define <16 x i8> @VectorCharConstantPool() {
; CHECK-LABEL: VectorCharConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI4_0@PCREL(0), 1
entry:
  ret <16 x i8> <i8 -128, i8 -127, i8 -126, i8 -125, i8 -124, i8 -123, i8 -122, i8 -121, i8 -120, i8 -119, i8 -118, i8 -117, i8 -116, i8 -115, i8 -114, i8 -113>
}

 define <8 x i16> @VectorShortConstantPool() {
; CHECK-LABEL: VectorShortConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI5_0@PCREL(0), 1
entry:
  ret <8 x i16> <i16 -32768, i16 -32767, i16 -32766, i16 -32765, i16 -32764, i16 -32763, i16 -32762, i16 -32761>
}

 define <4 x i32> @VectorIntConstantPool() {
; CHECK-LABEL: VectorIntConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI6_0@PCREL(0), 1
entry:
  ret <4 x i32> <i32 -2147483648, i32 -2147483647, i32 -2147483646, i32 -2147483645>
}

 define <2 x i64> @VectorLongLongConstantPool() {
; CHECK-LABEL: VectorLongLongConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI7_0@PCREL(0), 1
entry:
  ret <2 x i64> <i64 -9223372036854775808, i64 -9223372036854775807>
}

 define <1 x i128> @VectorInt128ConstantPool() {
; CHECK-LABEL: VectorInt128ConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI8_0@PCREL(0), 1
entry:
  ret <1 x i128> <i128 -27670116110564327424>
}

 define <4 x float> @VectorFloatConstantPool() {
; CHECK-LABEL: VectorFloatConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI9_0@PCREL(0), 1
entry:
  ret <4 x float> <float 0x380FFFF840000000, float 0x380FFF57C0000000, float 0x3843FFFB20000000, float 0x3843FF96C0000000>
}

 define <2 x double> @VectorDoubleConstantPool() {
; CHECK-LABEL: VectorDoubleConstantPool:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    plxv vs34, .LCPI10_0@PCREL(0), 1
entry:
  ret <2 x double> <double 2.225070e-308, double 2.225000e-308>
}
