; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -verify-machineinstrs -O3 \
; RUN:   -print-before=machine-combiner -print-after=machine-combiner -ppc-fma \
; RUN:    2>&1 | FileCheck %s

; REQUIRES: asserts

define double @fun0_fma2_add(ptr %x, double %A, double %B) {
; CHECK:      # *** IR Dump Before Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun0_fma2_add: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d, $f0d, $f2d
; CHECK-NEXT: [[Y:%2:fp64bit]] = COPY $f2d
; CHECK-NEXT: [[X:%1:fp64bit]] = COPY $f0d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %5:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %6:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: %7:vr64bit = {{.*}} WFADB_CCPseudo [[X]], [[Y]]
; CHECK-NEXT: %8:vr64bit = {{.*}} WFMADB killed [[M21:%3:vr64bit]], killed [[M22:%4:vr64bit]], killed %7:vr64bit
; CHECK-NEXT: %9:vr64bit = {{.*}} WFMADB killed [[M31:%5:vr64bit]], killed [[M32:%6:vr64bit]], killed %8:vr64bit
; CHECK-NEXT: $f0d = COPY %9:vr64bit
; CHECK-NEXT: Return implicit $f0d

; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun0_fma2_add: IsSSA, TracksLiveness
; CHECK:      %10:vr64bit = {{.*}} WFMADB killed [[M21]], killed [[M22]], [[X]]
; CHECK-NEXT: %11:vr64bit = {{.*}} WFMADB killed [[M31]], killed [[M32]], [[Y]]
; CHECK-NEXT: %9:vr64bit = {{.*}} WFADB_CCPseudo %10:vr64bit, %11:vr64bit
; CHECK-NEXT: $f0d = COPY %9:vr64bit
; CHECK-NEXT: Return implicit $f0d
entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4

  %mul1 = fmul reassoc nsz contract double %0, %1
  %mul2 = fmul reassoc nsz contract double %2, %3

  %A1 = fadd reassoc nsz contract double %A, %B
  %A2 = fadd reassoc nsz contract double %A1, %mul1
  %A3 = fadd reassoc nsz contract double %A2, %mul2

  ret double %A3
}

; Same as above, but with a long-latency factor in the root FMA which makes
; this undesirable.
define double @fun1_fma2_add_divop(ptr %x, double %A, double %B) {
; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun1_fma2_add_divop: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d, $f0d, $f2d
; CHECK-NEXT: %2:fp64bit = COPY $f2d
; CHECK-NEXT: %1:fp64bit = COPY $f0d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %5:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %6:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: %7:vr64bit = nofpexcept WFDDB %5:vr64bit, killed %6:vr64bit, implicit $fpc
; CHECK-NEXT: %8:vr64bit = {{.*}} WFADB_CCPseudo %1:fp64bit, %2:fp64bit
; CHECK-NEXT: %9:vr64bit = {{.*}} WFMADB killed %3:vr64bit, killed %4:vr64bit, killed %8:vr64bit
; CHECK-NEXT: %10:vr64bit = {{.*}} WFMADB %5:vr64bit, killed %7:vr64bit, killed %9:vr64bit
; CHECK-NEXT: $f0d = COPY %10:vr64bit
; CHECK-NEXT: Return implicit $f0d
entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4
  %div = fdiv double %2, %3

  %mul1 = fmul reassoc nsz contract double %0, %1
  %mul2 = fmul reassoc nsz contract double %2, %div

  %A1 = fadd reassoc nsz contract double %A, %B
  %A2 = fadd reassoc nsz contract double %A1, %mul1
  %A3 = fadd reassoc nsz contract double %A2, %mul2

  ret double %A3
}
