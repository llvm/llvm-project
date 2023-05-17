; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=-neon < %s

; The DAG combiner decided to use a vector load/store for this struct copy
; previously. This probably shouldn't happen without NEON, but the most
; important thing is that it compiles.

define void @store_combine() nounwind {
  %src = alloca { double, double }, align 8
  %dst = alloca { double, double }, align 8

  %src.realp = getelementptr inbounds { double, double }, ptr %src, i32 0, i32 0
  %src.real = load double, ptr %src.realp
  %src.imagp = getelementptr inbounds { double, double }, ptr %src, i32 0, i32 1
  %src.imag = load double, ptr %src.imagp

  %dst.realp = getelementptr inbounds { double, double }, ptr %dst, i32 0, i32 0
  %dst.imagp = getelementptr inbounds { double, double }, ptr %dst, i32 0, i32 1
  store double %src.real, ptr %dst.realp
  store double %src.imag, ptr %dst.imagp
  ret void
}
