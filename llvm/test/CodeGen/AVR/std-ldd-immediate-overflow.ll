; RUN: llc -O0 < %s -march=avr | FileCheck %s

define i32 @std_ldd_overflow() {
  %src = alloca [4 x i8]
  %dst = alloca [4 x i8]
  %buf = alloca [28 x i16]
  %1 = bitcast [4 x i8]* %src to i32*
  store i32 0, i32 *%1
  %2 = bitcast [4 x i8]* %dst to i8*
  %3 = bitcast [4 x i8]* %src to i8*
  call void @llvm.memcpy.p0i8.p0i8.i16(i8* %2, i8* %3, i16 4, i1 false)
; CHECK-NOT: std {{[XYZ]}}+64, {{r[0-9]+}}
; CHECK-NOT: ldd {{r[0-9]+}}, {{[XYZ]}}+64

  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i16(i8* nocapture writeonly, i8* nocapture readonly, i16, i1)
