; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z16 -O3 -print-before=peephole-opt \
; RUN:   -print-after=peephole-opt -verify-machineinstrs 2>&1 | FileCheck %s

define void @f0(i64 %a, i64 %b, float %f1, ptr %src1, ptr %src2, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %6:vr32bit = VL32 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s32) from %ir.src1)
; CHECK-NEXT:  nofpexcept WFCSB %2:fp32bit, killed %6:vr32bit, implicit-def $cc, implicit $fpc
; CHECK:       %9:vr32bit = VL32 %4:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  nofpexcept WFCSB %2:fp32bit, %9:vr32bit, implicit-def $cc, implicit $fpc
; CHECK:       VST32 %9:vr32bit, %5:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       nofpexcept CEB %2:fp32bit, [[ADDR1]], 0, $noreg, implicit-def $cc, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       nofpexcept WFCSB %2:fp32bit, %9:vr32bit, implicit-def $cc, implicit $fpc

  %L1 = load float, ptr %src1
  %C1 = fcmp oeq float %f1, %L1
  %S1 = select i1 %C1, i64 0, i64 1
  store volatile i64 %S1, ptr %dst

  %L2 = load float, ptr %src2
  %C2 = fcmp oeq float %f1, %L2
  %S2 = select i1 %C2, i64 0, i64 1
  store volatile i64 %S2, ptr %dst
  store volatile float %L2, ptr %dst

  ret void
}

define void @f1(i64 %a, i64 %b, double %f1, ptr %src1, ptr %src2, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %6:vr64bit = VL64 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s64) from %ir.src1)
; CHECK-NEXT:  nofpexcept WFCDB %2:fp64bit, killed %6:vr64bit, implicit-def $cc, implicit $fpc
; CHECK:       %9:vr64bit = VL64 %4:addr64bit, 0, $noreg :: (load (s64) from %ir.src2)
; CHECK-NEXT:  nofpexcept WFCDB %2:fp64bit, %9:vr64bit, implicit-def $cc, implicit $fpc
; CHECK:       VST64 %9:vr64bit, %5:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       nofpexcept CDB %2:fp64bit, [[ADDR1]], 0, $noreg, implicit-def $cc, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       nofpexcept WFCDB %2:fp64bit, %9:vr64bit, implicit-def $cc, implicit $fpc

  %L1 = load double, ptr %src1
  %C1 = fcmp oeq double %f1, %L1
  %S1 = select i1 %C1, i64 0, i64 1
  store volatile i64 %S1, ptr %dst

  %L2 = load double, ptr %src2
  %C2 = fcmp oeq double %f1, %L2
  %S2 = select i1 %C2, i64 0, i64 1
  store volatile i64 %S2, ptr %dst
  store volatile double %L2, ptr %dst

  ret void
}
