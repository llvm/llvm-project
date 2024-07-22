; RUN: llc -O0 < %s -march=avr | FileCheck %s

define i32 @std_ldd_overflow() {
  %src = alloca [4 x i8]
  %dst = alloca [4 x i8]
  %buf = alloca [28 x i16]
  %1 = bitcast ptr %src to ptr
  store i32 0, ptr %1
  %2 = bitcast ptr %dst to ptr
  %3 = bitcast ptr %src to ptr
  call void @llvm.memcpy.p0.p0.i16(ptr %2, ptr %3, i16 4, i1 false)
; CHECK-NOT: std {{[XYZ]}}+64, {{r[0-9]+}}
; CHECK-NOT: ldd {{r[0-9]+}}, {{[XYZ]}}+64

  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i16(ptr nocapture writeonly, ptr nocapture readonly, i16, i1)
