; RUN: opt -mtriple i686-windows-itanium -O2 -o - %s | llvm-dis | FileCheck %s

target triple = "i686-windows-itanium"

declare dllimport double @floor(double)

define dllexport float @test(float %f) {
  %conv = fpext float %f to double
  %call = tail call double @floor(double %conv)
  %cast = fptrunc double %call to float
  ret float %cast
}

; CHECK-NOT: floorf
; CHECK: floor

