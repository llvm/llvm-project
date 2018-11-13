; RUN: llc -o - %s -mtriple=mips-unknown-linux-gnu \
; RUN:     -mcpu=mips32 -mattr=+fpxx \
; RUN:     -stop-after=expand-isel-pseudos | \
; RUN:     FileCheck %s -check-prefix=FPXX-IMPLICIT-SP

; RUN: llc -o - %s -mtriple=mips-unknown-linux-gnu \
; RUN:     -mcpu=mips32r6 -mattr=+fp64,+nooddspreg \
; RUN:     -stop-after=expand-isel-pseudos | \
; RUN:     FileCheck %s -check-prefix=FP64-IMPLICIT-SP

; RUN: llc -o - %s -mtriple=mips-unknown-linux-gnu \
; RUN:     -mcpu=mips32r2 -mattr=+fpxx \
; RUN:     -stop-after=expand-isel-pseudos | \
; RUN:     FileCheck %s -check-prefix=NO-IMPLICIT-SP

define double @foo2(i32 signext %v1, double %d1) {
entry:
; FPXX-IMPLICIT-SP: BuildPairF64 %{{[0-9]+}}, %{{[0-9]+}}, implicit $sp
; FPXX-IMPLICIT-SP: ExtractElementF64 killed %{{[0-9]+}}, 1, implicit $sp
; FP64-IMPLICIT-SP: BuildPairF64_64 %{{[0-9]+}}, %{{[0-9]+}}, implicit $sp
; FP64-IMPLICIT-SP: ExtractElementF64_64 killed %{{[0-9]+}}, 1, implicit $sp
; NO-IMPLICIT-SP: BuildPairF64 %{{[0-9]+}}, %{{[0-9]+}}
; NO-IMPLICIT-SP-NOT: BuildPairF64 %{{[0-9]+}}, %{{[0-9]+}}, implicit $sp
; NO-IMPLICIT-SP: ExtractElementF64 killed %{{[0-9]+}}, 1
; NO-IMPLICIT-SP-NOT: ExtractElementF64 killed %{{[0-9]+}}, 1, implicit $sp
  %conv = fptrunc double %d1 to float
  %0 = tail call float @llvm.copysign.f32(float 1.000000e+00, float %conv)
  %conv1 = fpext float %0 to double
  ret double %conv1
}

declare float @llvm.copysign.f32(float, float)
