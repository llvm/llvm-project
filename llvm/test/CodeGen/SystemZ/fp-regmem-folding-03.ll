; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z16 -O3 -print-before=peephole-opt \
; RUN:   -print-after=peephole-opt -verify-machineinstrs 2>&1 | FileCheck %s

define void @f0(ptr %src1, ptr %src2, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:vr32bit = VL32 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s32) from %ir.src1)
; CHECK-NEXT:  %4:vr64bit = nofpexcept WLDEB killed %3:vr32bit, implicit $fpc
; CHECK:       %5:vr32bit = VL32 %1:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  %6:vr64bit = nofpexcept WLDEB %5:vr32bit, implicit $fpc

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %4:fp64bit = nofpexcept LDEB [[ADDR1]], 0, $noreg, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %5:vr32bit = VL32 %1:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  %6:vr64bit = nofpexcept WLDEB %5:vr32bit, implicit $fpc

  %L1 = load float, ptr %src1
  %D1 = fpext float %L1 to double
  store volatile double %D1, ptr %dst

  %L2 = load float, ptr %src2
  %D2 = fpext float %L2 to double
  store volatile double %D2, ptr %dst
  store volatile float %L2, ptr %dst

  ret void
}

define void @f1(ptr %ptr, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %2:vr32bit = VL32 [[ADDR2:%0:addr64bit]], 0, $noreg :: (load (s32) from %ir.ptr)
; CHECK-NEXT:  %3:vr32bit = nofpexcept WFSQSB killed %2:vr32bit, implicit $fpc
; CHECK:       %4:vr32bit = VL32 %0:addr64bit, 0, $noreg :: (load (s32) from %ir.ptr)
; CHECK-NEXT:  %5:vr32bit = nofpexcept WFSQSB %4:vr32bit, implicit $fpc

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:fp32bit = nofpexcept SQEB [[ADDR2]], 0, $noreg, implicit $fpc :: (load (s32) from %ir.ptr)
; CHECK:       %4:vr32bit = VL32 %0:addr64bit, 0, $noreg :: (load (s32) from %ir.ptr)
; CHECK-NEXT:  %5:vr32bit = nofpexcept WFSQSB %4:vr32bit, implicit $fpc

  %L1 = load float, ptr %ptr
  %S1 = call float @llvm.sqrt.f32(float %L1)
  store volatile float %S1, ptr %dst

  %L2 = load float, ptr %ptr
  %S2 = call float @llvm.sqrt.f32(float %L2)
  store volatile float %S2, ptr %dst
  store volatile float %L2, ptr %dst

  ret void
}

define void @f2(ptr %ptr, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %2:vr64bit = VL64 [[ADDR2:%0:addr64bit]], 0, $noreg :: (load (s64) from %ir.ptr)
; CHECK-NEXT:  %3:vr64bit = nofpexcept WFSQDB killed %2:vr64bit, implicit $fpc
; CHECK:       %4:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.ptr)
; CHECK-NEXT:  %5:vr64bit = nofpexcept WFSQDB %4:vr64bit, implicit $fpc

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:fp64bit = nofpexcept SQDB [[ADDR2]], 0, $noreg, implicit $fpc :: (load (s64) from %ir.ptr)
; CHECK:        %4:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.ptr)
; CHECK-NEXT:  %5:vr64bit = nofpexcept WFSQDB %4:vr64bit, implicit $fpc

  %L1 = load double, ptr %ptr
  %S1 = call double @llvm.sqrt.f64(double %L1)
  store volatile double %S1, ptr %dst

  %L2 = load double, ptr %ptr
  %S2 = call double @llvm.sqrt.f64(double %L2)
  store volatile double %S2, ptr %dst
  store volatile double %L2, ptr %dst

  ret void
}
