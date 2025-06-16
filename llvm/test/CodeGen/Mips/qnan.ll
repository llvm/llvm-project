; RUN: llc -O3 -mcpu=mips32r2 -mtriple=mips-linux-gnu < %s -o - | FileCheck %s -check-prefixes=MIPS_Legacy
; RUN: llc -O3 -mcpu=mips32r2 -mtriple=mips-linux-gnu -mattr=+nan2008 < %s -o - | FileCheck %s -check-prefixes=MIPS_NaN2008

define dso_local float @nan(float noundef %a, float noundef %b) local_unnamed_addr #0 {
; MIPS_Legacy: $CPI0_0:
; MIPS_Legacy-NEXT: .4byte  0x7fa00000 # float NaN

; MIPS_NaN2008: $CPI0_0:
; MIPS_NaN2008-NEXT: .4byte  0x7fc00000 # float NaN

entry:
  %0 = tail call float @llvm.minimum.f32(float %a, float %b)
  ret float %0
}
