; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check for successful compilation
; CHECK: jumpr r31

target triple = "hexagon"

; Function Attrs: nounwind
define hidden fastcc void @f0() unnamed_addr #0 {
b0:
  call void @llvm.va_copy(ptr nonnull undef, ptr nonnull undef)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_copy(ptr, ptr) #0

attributes #0 = { nounwind }
