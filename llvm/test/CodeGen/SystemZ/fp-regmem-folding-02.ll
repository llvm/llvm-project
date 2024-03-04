; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z16 -O3 -print-before=peephole-opt \
; RUN:   -print-after=peephole-opt -verify-machineinstrs 2>&1 | FileCheck %s

define void @f0(float %A, ptr %src, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:vr32bit = VL32 [[ADDR1:%[0-9]+:addr64bit]], 4, $noreg :: (load (s32) from %ir.arrayidx1)
; CHECK-NEXT:  %4:vr32bit = VL32 %1:addr64bit, 8, $noreg :: (load (s32) from %ir.arrayidx2)
; CHECK-NEXT:  vr32bit = contract nofpexcept WFMASB killed %3:vr32bit, killed %4:vr32bit, %0:fp32bit, implicit $fpc
; CHECK:       %6:vr32bit = VL32 %1:addr64bit, 12, $noreg :: (load (s32) from %ir.arrayidx3)
; CHECK-NEXT:  %7:vr32bit = VL32 %1:addr64bit, 16, $noreg :: (load (s32) from %ir.arrayidx4)
; CHECK-NEXT:  %8:vr32bit = contract nofpexcept WFMASB %6:vr32bit, %7:vr32bit, %0:fp32bit, implicit $fpc
; CHECK-NEXT:  VST32 killed %8:vr32bit, %2:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)
; CHECK-NEXT:  VST32 %6:vr32bit, %2:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)
; CHECK-NEXT:  VST32 %7:vr32bit, %2:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept MAEB %0:fp32bit(tied-def 0), killed %4:fp32bit, [[ADDR1]], 4, $noreg, implicit $fpc :: (load (s32) from %ir.arrayidx1)
; CHECK:       vr32bit = contract nofpexcept WFMASB %6:vr32bit, %7:vr32bit, %0:fp32bit, implicit $fpc

  %arrayidx1 = getelementptr inbounds float, ptr %src, i64 1
  %arrayidx2 = getelementptr inbounds float, ptr %src, i64 2
  %L1l = load float, ptr %arrayidx1
  %L1r = load float, ptr %arrayidx2
  %M1 = fmul contract float %L1l, %L1r
  %A1 = fadd contract float %A, %M1
  store volatile float %A1, ptr %dst

  %arrayidx3 = getelementptr inbounds float, ptr %src, i64 3
  %arrayidx4 = getelementptr inbounds float, ptr %src, i64 4
  %L2l = load float, ptr %arrayidx3
  %L2r = load float, ptr %arrayidx4
  %M2 = fmul contract float %L2l, %L2r
  %A2 = fadd contract float %A, %M2
  store volatile float %A2, ptr %dst
  store volatile float %L2l, ptr %dst
  store volatile float %L2r, ptr %dst

  ret void
}

define void @f1(double %A, ptr %src, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:vr64bit = VL64 [[ADDR1:%[0-9]+:addr64bit]], 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT:  %4:vr64bit = VL64 %1:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT:  vr64bit = contract nofpexcept WFMADB killed %3:vr64bit, killed %4:vr64bit, %0:fp64bit, implicit $fpc
; CHECK:       %6:vr64bit = VL64 %1:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx3)
; CHECK-NEXT:  %7:vr64bit = VL64 %1:addr64bit, 32, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT:  %8:vr64bit = contract nofpexcept WFMADB %6:vr64bit, %7:vr64bit, %0:fp64bit, implicit $fpc
; CHECK-NEXT:  VST64 killed %8:vr64bit, %2:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)
; CHECK-NEXT:  VST64 %6:vr64bit, %2:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)
; CHECK-NEXT:  VST64 %7:vr64bit, %2:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept MADB %0:fp64bit(tied-def 0), killed %4:fp64bit, [[ADDR1]], 8, $noreg, implicit $fpc :: (load (s64) from %ir.arrayidx1)
; CHECK:       vr64bit = contract nofpexcept WFMADB %6:vr64bit, %7:vr64bit, %0:fp64bit, implicit $fpc

  %arrayidx1 = getelementptr inbounds double, ptr %src, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %src, i64 2
  %L1l = load double, ptr %arrayidx1
  %L1r = load double, ptr %arrayidx2
  %M1 = fmul contract double %L1l, %L1r
  %A1 = fadd contract double %A, %M1
  store volatile double %A1, ptr %dst

  %arrayidx3 = getelementptr inbounds double, ptr %src, i64 3
  %arrayidx4 = getelementptr inbounds double, ptr %src, i64 4
  %L2l = load double, ptr %arrayidx3
  %L2r = load double, ptr %arrayidx4
  %M2 = fmul contract double %L2l, %L2r
  %A2 = fadd contract double %A, %M2
  store volatile double %A2, ptr %dst
  store volatile double %L2l, ptr %dst
  store volatile double %L2r, ptr %dst

  ret void
}

define void @f2(float %A, ptr %src, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:vr32bit = VL32 [[ADDR1:%[0-9]+:addr64bit]], 4, $noreg :: (load (s32) from %ir.arrayidx1)
; CHECK-NEXT:  %4:vr32bit = VL32 %1:addr64bit, 8, $noreg :: (load (s32) from %ir.arrayidx2)
; CHECK-NEXT:  vr32bit = nofpexcept WFMSSB killed %3:vr32bit, killed %4:vr32bit, %0:fp32bit, implicit $fpc
; CHECK:       %6:vr32bit = VL32 %1:addr64bit, 12, $noreg :: (load (s32) from %ir.arrayidx3)
; CHECK-NEXT:  %7:vr32bit = VL32 %1:addr64bit, 16, $noreg :: (load (s32) from %ir.arrayidx4)
; CHECK-NEXT:  %8:vr32bit = nofpexcept WFMSSB %6:vr32bit, %7:vr32bit, %0:fp32bit, implicit $fpc
; CHECK-NEXT:  VST32 killed %8:vr32bit, %2:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)
; CHECK-NEXT:  VST32 %6:vr32bit, %2:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)
; CHECK-NEXT:  VST32 %7:vr32bit, %2:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept MSEB %0:fp32bit(tied-def 0), killed %4:fp32bit, [[ADDR1]], 4, $noreg, implicit $fpc :: (load (s32) from %ir.arrayidx1)
; CHECK:       vr32bit = nofpexcept WFMSSB %6:vr32bit, %7:vr32bit, %0:fp32bit, implicit $fpc
  %arrayidx1 = getelementptr inbounds float, ptr %src, i64 1
  %arrayidx2 = getelementptr inbounds float, ptr %src, i64 2
  %L1l = load float, ptr %arrayidx1
  %L1r = load float, ptr %arrayidx2
  %Negacc1 = fneg float %A
  %A1 = call float @llvm.fma.f32 (float %L1l, float %L1r, float %Negacc1)
  store volatile float %A1, ptr %dst

  %arrayidx3 = getelementptr inbounds float, ptr %src, i64 3
  %arrayidx4 = getelementptr inbounds float, ptr %src, i64 4
  %L2l = load float, ptr %arrayidx3
  %L2r = load float, ptr %arrayidx4
  %Negacc2 = fneg float %A
  %A2 = call float @llvm.fma.f32 (float %L2l, float %L2r, float %Negacc2)
  store volatile float %A2, ptr %dst
  store volatile float %L2l, ptr %dst
  store volatile float %L2r, ptr %dst

  ret void
}

define void @f3(double %A, ptr %src, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f3: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       %3:vr64bit = VL64 [[ADDR1:%[0-9]+:addr64bit]], 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT:  %4:vr64bit = VL64 %1:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT:  vr64bit = nofpexcept WFMSDB killed %3:vr64bit, killed %4:vr64bit, %0:fp64bit, implicit $fpc
; CHECK:       %6:vr64bit = VL64 %1:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx3)
; CHECK-NEXT:  %7:vr64bit = VL64 %1:addr64bit, 32, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT:  %8:vr64bit = nofpexcept WFMSDB %6:vr64bit, %7:vr64bit, %0:fp64bit, implicit $fpc
; CHECK-NEXT:  VST64 killed %8:vr64bit, %2:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)
; CHECK-NEXT:  VST64 %6:vr64bit, %2:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)
; CHECK-NEXT:  VST64 %7:vr64bit, %2:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f3: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept MSDB %0:fp64bit(tied-def 0), killed %4:fp64bit, [[ADDR1]], 8, $noreg, implicit $fpc :: (load (s64) from %ir.arrayidx1)
; CHECK:       vr64bit = nofpexcept WFMSDB %6:vr64bit, %7:vr64bit, %0:fp64bit, implicit $fpc
  %arrayidx1 = getelementptr inbounds double, ptr %src, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %src, i64 2
  %L1l = load double, ptr %arrayidx1
  %L1r = load double, ptr %arrayidx2
  %Negacc1 = fneg double %A
  %A1 = call double @llvm.fma.f64 (double %L1l, double %L1r, double %Negacc1)
  store volatile double %A1, ptr %dst

  %arrayidx3 = getelementptr inbounds double, ptr %src, i64 3
  %arrayidx4 = getelementptr inbounds double, ptr %src, i64 4
  %L2l = load double, ptr %arrayidx3
  %L2r = load double, ptr %arrayidx4
  %Negacc2 = fneg double %A
  %A2 = call double @llvm.fma.f64 (double %L2l, double %L2r, double %Negacc2)
  store volatile double %A2, ptr %dst
  store volatile double %L2l, ptr %dst
  store volatile double %L2r, ptr %dst

  ret void
}
