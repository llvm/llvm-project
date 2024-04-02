; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -verify-machineinstrs -O3 \
; RUN:   -print-before=machine-combiner -print-after=machine-combiner -z-fma \
; RUN:    2>&1 | FileCheck %s
; REQUIRES: asserts

; Test reassociation involving fma.

; The incoming accumulator is stalling so it is worth putting the
; multiplications in parallell with it.
define double @fun0_fma2_divop(ptr %x) {
; CHECK:      # *** IR Dump Before Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun0_fma2_divop: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: [[M21:%1:vr64bit]] = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: [[M22:%2:vr64bit]] = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: [[M11:%3:vr64bit]] = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: [[M12:%4:vr64bit]] = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: [[DIV:%5:vr64bit]] = nofpexcept WFDDB %3:vr64bit, %4:vr64bit, implicit $fpc
; CHECK-NEXT: %6:vr64bit = {{.*}} WFMADB_CCPseudo killed [[M21]], killed [[M22]], killed [[DIV]]
; CHECK-NEXT: %7:vr64bit = {{.*}} WFMADB_CCPseudo        [[M11]],        [[M12]], killed %6:vr64bit
; CHECK-NEXT: $f0d = COPY %7:vr64bit
; CHECK-NEXT: Return implicit $f0d

; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun0_fma2_divop: IsSSA, TracksLiveness
; CHECK:      %8:vr64bit = {{.*}} WFMDB killed [[M21]], killed [[M22]]
; CHECK-NEXT: %9:vr64bit = {{.*}} WFMADB_CCPseudo [[M11]], [[M12]], %8:vr64bit
; CHECK-NEXT: %7:vr64bit = {{.*}} WFADB_CCPseudo killed [[DIV]], %9:vr64bit
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
  %mul2 = fmul reassoc nsz contract double %2, %3

  %A1 = fadd reassoc nsz contract double %div, %mul1
  %A2 = fadd reassoc nsz contract double %A1, %mul2

  ret double %A2
}

; The non-profitable case:
define double @fun1_fma2(ptr %x, double %Arg) {
; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun1_fma2: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d, $f0d
; CHECK-NEXT: %1:fp64bit = COPY $f0d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: %2:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %5:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: %6:vr64bit = {{.*}} WFMADB_CCPseudo killed %2:vr64bit, killed %3:vr64bit, %1:fp64bit
; CHECK-NEXT: %7:vr64bit = {{.*}} WFMADB_CCPseudo killed %4:vr64bit, killed %5:vr64bit, killed %6:vr64bit
; CHECK-NEXT: $f0d = COPY %7:vr64bit
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

  %A1 = fadd reassoc nsz contract double %Arg, %mul1
  %A2 = fadd reassoc nsz contract double %A1, %mul2

  ret double %A2
}

; Keep the two FMAs, but change order due to the long latency divide.
define double @fun2_fma2(ptr %x) {
; CHECK:      # *** IR Dump Before Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun2_fma2: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: %1:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: %2:vr64bit = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: [[DIV:%5:vr64bit]] = nofpexcept WFDDB %3:vr64bit, %4:vr64bit, implicit $fpc
; CHECK-NEXT: %6:vr64bit = {{.*}} WFMADB_CCPseudo killed %1:vr64bit, killed [[DIV]], killed %2:vr64bit
; CHECK-NEXT: %7:vr64bit = {{.*}} WFMADB_CCPseudo %3:vr64bit, %4:vr64bit, killed %6:vr64bit

; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun2_fma2: IsSSA, TracksLiveness
; CHECK:      %12:vr64bit = {{.*}} WFMADB_CCPseudo %3:vr64bit, %4:vr64bit, killed %2:vr64bit
; CHECK-NEXT: %7:vr64bit = {{.*}} WFMADB_CCPseudo killed %1:vr64bit, killed [[DIV]], %12:vr64bit

entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4
  %div = fdiv double %2, %3

  %mul1 = fmul reassoc nsz contract double %0, %div
  %mul2 = fmul reassoc nsz contract double %2, %3

  %A1 = fadd reassoc nsz contract double %1, %mul1
  %A2 = fadd reassoc nsz contract double %A1, %mul2

  ret double %A2
}
