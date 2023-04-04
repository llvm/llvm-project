; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -verify-machineinstrs -O3 \
; RUN:   -print-before=machine-combiner -print-after=machine-combiner -z-fma \
; RUN:    2>&1 | FileCheck %s

; REQUIRES: asserts

; No improvement possible.
define double @fun0_fma1add(ptr %x) {
; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun0_fma1add: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: %1:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: %2:vr64bit = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: %5:vr64bit = {{.*}} WFADB_CCPseudo killed %3:vr64bit, killed %4:vr64bit
; CHECK-NEXT: %6:vr64bit = {{.*}} WFMADB killed %1:vr64bit, killed %2:vr64bit, killed %5:vr64bit
; CHECK-NEXT: $f0d = COPY %6:vr64bit
; CHECK-NEXT: Return implicit $f0d
entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4

  %mul = fmul reassoc nsz contract double %0, %1

  %A1 = fadd reassoc nsz contract double %2, %3
  %A2 = fadd reassoc nsz contract double %A1, %mul

  ret double %A2
}

; The RHS of the Add is stalling, so move up the FMA to the LHS.
define double @fun1_fma1add_divop(ptr %x) {
; CHECK:      # *** IR Dump Before Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun1_fma1add_divop: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: [[M21:%1:vr64bit]] = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: [[M22:%2:vr64bit]] = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: [[T1:%3:vr64bit]] = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: [[DIV:%5:vr64bit]] = nofpexcept WFDDB [[T1]], killed %4:vr64bit, implicit $fpc
; CHECK-NEXT: %6:vr64bit = {{.*}} WFADB_CCPseudo [[T1]], killed [[DIV]]
; CHECK-NEXT: %7:vr64bit = {{.*}} WFMADB killed [[M21]], killed [[M22]], killed %6:vr64bit
; CHECK-NEXT: $f0d = COPY %7:vr64bit
; CHECK-NEXT: Return implicit $f0d

; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun1_fma1add_divop: IsSSA, TracksLiveness
; CHECK:      %8:vr64bit = {{.*}} WFMADB killed [[M21]], killed [[M22]], [[T1]]
; CHECK-NEXT: %7:vr64bit = {{.*}} WFADB_CCPseudo %8:vr64bit, killed [[DIV]]
entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4
  %div = fdiv double %2, %3

  %mul = fmul reassoc nsz contract double %0, %1

  %A1 = fadd reassoc nsz contract double %2, %div
  %A2 = fadd reassoc nsz contract double %A1, %mul

  ret double %A2
}

; The LHS of the Add is stalling, so move up the FMA to the RHS.
define double @fun2_fma1add_divop(ptr %x) {
; CHECK:      # *** IR Dump Before Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun2_fma1add_divop: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: [[M21:%1:vr64bit]] = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: [[M22:%2:vr64bit]] = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: [[T2:%4:vr64bit]] = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: [[DIV:%5:vr64bit]] = nofpexcept WFDDB killed %3:vr64bit, %4:vr64bit, implicit $fpc
; CHECK-NEXT: %6:vr64bit = {{.*}} WFADB_CCPseudo killed [[DIV]], [[T2]]
; CHECK-NEXT: %7:vr64bit = {{.*}} WFMADB killed [[M21]], killed [[M22]], killed %6:vr64bit

; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun2_fma1add_divop: IsSSA, TracksLiveness
; CHECK:      %9:vr64bit = {{.*}} WFMADB killed [[M21]], killed [[M22]], [[T2]]
; CHECK:      %7:vr64bit = {{.*}} WFADB_CCPseudo %9:vr64bit, killed [[DIV]]
entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4
  %div = fdiv double %2, %3

  %mul = fmul reassoc nsz contract double %0, %1

  %A1 = fadd reassoc nsz contract double %div, %3
  %A2 = fadd reassoc nsz contract double %A1, %mul

  ret double %A2
}
