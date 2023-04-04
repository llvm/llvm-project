; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -O3 -print-before=machine-combiner \
; RUN:    -print-after=machine-combiner -debug-only=machine-combiner,systemz-II -z-fma 2>&1 \
; RUN:    | FileCheck %s

; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -O3 \
; RUN:    -print-after=machine-combiner -debug-only=machine-combiner,systemz-II -ppc-fma 2>&1 \
; RUN:    | FileCheck %s --check-prefix=ALT

; REQUIRES: asserts

; Test transformation of a sequence of 8 FMAs, with different patterns.

define double @fun_fma8(ptr %x, double %A) {
; CHECK:      # *** IR Dump Before Machine InstCombiner (machine-combiner) ***:
; CHECK-NEXT: # Machine code for function fun_fma8: IsSSA, TracksLiveness
; CHECK:      bb.0.entry:
; CHECK-NEXT: liveins: $r2d, $f0d
; CHECK-NEXT: %1:fp64bit = COPY $f0d
; CHECK-NEXT: %0:addr64bit = COPY $r2d
; CHECK-NEXT: %2:vr64bit = VL64 %0:addr64bit, 0, $noreg :: (load (s64) from %ir.x)
; CHECK-NEXT: %3:vr64bit = VL64 %0:addr64bit, 8, $noreg :: (load (s64) from %ir.arrayidx1)
; CHECK-NEXT: %4:vr64bit = VL64 %0:addr64bit, 16, $noreg :: (load (s64) from %ir.arrayidx2)
; CHECK-NEXT: %5:vr64bit = VL64 %0:addr64bit, 24, $noreg :: (load (s64) from %ir.arrayidx4)
; CHECK-NEXT: %6:vr64bit = VL64 %0:addr64bit, 32, $noreg :: (load (s64) from %ir.arrayidx6)
; CHECK-NEXT: %7:vr64bit = VL64 %0:addr64bit, 40, $noreg :: (load (s64) from %ir.arrayidx8)
; CHECK-NEXT: %8:vr64bit = VL64 %0:addr64bit, 48, $noreg :: (load (s64) from %ir.arrayidx10)
; CHECK-NEXT: %9:vr64bit = VL64 %0:addr64bit, 56, $noreg :: (load (s64) from %ir.arrayidx12)
; CHECK-NEXT: %10:vr64bit = VL64 %0:addr64bit, 64, $noreg :: (load (s64) from %ir.arrayidx14)
; CHECK-NEXT: %11:vr64bit = VL64 %0:addr64bit, 72, $noreg :: (load (s64) from %ir.arrayidx16)
; CHECK-NEXT: %12:vr64bit = VL64 %0:addr64bit, 80, $noreg :: (load (s64) from %ir.arrayidx18)
; CHECK-NEXT: %13:vr64bit = VL64 %0:addr64bit, 88, $noreg :: (load (s64) from %ir.arrayidx20)
; CHECK-NEXT: %14:vr64bit = VL64 %0:addr64bit, 96, $noreg :: (load (s64) from %ir.arrayidx22)
; CHECK-NEXT: %15:vr64bit = VL64 %0:addr64bit, 104, $noreg :: (load (s64) from %ir.arrayidx24)
; CHECK-NEXT: %16:vr64bit = VL64 %0:addr64bit, 112, $noreg :: (load (s64) from %ir.arrayidx26)
; CHECK-NEXT: %17:vr64bit = VL64 %0:addr64bit, 120, $noreg :: (load (s64) from %ir.arrayidx28)
; CHECK-NEXT: %18:vr64bit = {{.*}} WFMADB killed %2:vr64bit, killed %3:vr64bit, %1:fp64bit
; CHECK-NEXT: %19:vr64bit = {{.*}} WFMADB killed %4:vr64bit, killed %5:vr64bit, killed %18:vr64bit
; CHECK-NEXT: %20:vr64bit = {{.*}} WFMADB killed %6:vr64bit, killed %7:vr64bit, killed %19:vr64bit
; CHECK-NEXT: %21:vr64bit = {{.*}} WFMADB killed %8:vr64bit, killed %9:vr64bit, killed %20:vr64bit
; CHECK-NEXT: %22:vr64bit = {{.*}} WFMADB killed %10:vr64bit, killed %11:vr64bit, killed %21:vr64bit
; CHECK-NEXT: %23:vr64bit = {{.*}} WFMADB killed %12:vr64bit, killed %13:vr64bit, killed %22:vr64bit
; CHECK-NEXT: %24:vr64bit = {{.*}} WFMADB killed %14:vr64bit, killed %15:vr64bit, killed %23:vr64bit
; CHECK-NEXT: %25:vr64bit = {{.*}} WFMADB killed %16:vr64bit, killed %17:vr64bit, killed %24:vr64bit
; CHECK-NEXT: $f0d = COPY %25:vr64bit
; CHECK-NEXT: Return implicit $f0d

; CHECK:      Machine InstCombiner: fun_fma8
; CHECK:      add pattern FMA2_P1P0
; CHECK-NEXT: add pattern FMA2_P0P1
; CHECK-NEXT: add pattern FMA2
; CHECK:      reassociating using pattern FMA_P1P0
; CHECK:        Dependence data for %21:vr64bit = {{.*}} WFMADB
; CHECK-NEXT: 	NewRootDepth: 16	RootDepth: 22	It MustReduceDepth 	  and it does it
; CHECK-NEXT:  		Resource length before replacement: 16 and after: 16
; CHECK-NEXT: 		  As result it IMPROVES/PRESERVES Resource Length
; CHECK:      add pattern FMA2_P1P0
; CHECK-NEXT: add pattern FMA2_P0P1
; CHECK-NEXT: add pattern FMA2
; CHECK-NEXT: reassociating using pattern FMA_P1P0
; CHECK-NEXT:   Dependence data for %23:vr64bit = {{.*}} WFMADB
; CHECK-NEXT: 	NewRootDepth: 22	RootDepth: 28	It MustReduceDepth 	  and it does it
; CHECK:      		Resource length before replacement: 16 and after: 16
; CHECK-NEXT: 		  As result it IMPROVES/PRESERVES Resource Length
; CHECK-NEXT: add pattern FMA1_Add_L
; CHECK-NEXT: add pattern FMA1_Add_R
; CHECK-NEXT: reassociating using pattern FMA1_Add_L
; CHECK-NEXT:   Dependence data for %24:vr64bit = {{.*}} WFMADB
; CHECK-NEXT: 	NewRootDepth: 28	RootDepth: 28	It MustReduceDepth 	  but it does NOT do it
; CHECK-NEXT: reassociating using pattern FMA1_Add_R
; CHECK-NEXT:   Dependence data for %24:vr64bit = {{.*}} WFMADB
; CHECK-NEXT: 	NewRootDepth: 22	RootDepth: 28	It MustReduceDepth 	  and it does it
; CHECK-NEXT: 		Resource length before replacement: 16 and after: 16
; CHECK-NEXT: 		  As result it IMPROVES/PRESERVES Resource Length

; CHECK:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; CHECK:      %18:vr64bit = {{.*}} WFMADB killed %2:vr64bit, killed %3:vr64bit, %1:fp64bit
; CHECK-NEXT: %19:vr64bit = {{.*}} WFMADB killed %4:vr64bit, killed %5:vr64bit, killed %18:vr64bit
; CHECK-NEXT: %36:vr64bit = {{.*}} WFMDB killed %6:vr64bit, killed %7:vr64bit
; CHECK-NEXT: %37:vr64bit = {{.*}} WFMADB killed %8:vr64bit, killed %9:vr64bit, %36:vr64bit
; CHECK-NEXT: %21:vr64bit = {{.*}} WFADB_CCPseudo killed %19:vr64bit, %37:vr64bit
; CHECK-NEXT: %40:vr64bit = {{.*}} WFMDB killed %10:vr64bit, killed %11:vr64bit
; CHECK-NEXT: %41:vr64bit = {{.*}} WFMADB killed %12:vr64bit, killed %13:vr64bit, %40:vr64bit
; CHECK-NEXT: %43:vr64bit = {{.*}} WFMADB killed %14:vr64bit, killed %15:vr64bit, %41:vr64bit
; CHECK-NEXT: %24:vr64bit = {{.*}} WFADB_CCPseudo %43:vr64bit, killed %21:vr64bit
; CHECK-NEXT: %25:vr64bit = {{.*}} WFMADB killed %16:vr64bit, killed %17:vr64bit, killed %24:vr64bit

; ALT:       Machine InstCombiner: fun_fma8
; ALT-NEXT: Combining MBB entry
; ALT-NEXT: add pattern FMA3
; ALT-NEXT: reassociating using pattern FMA3
; ALT-NEXT:   Dependence data for %20:vr64bit = {{.*}} WFMADB
; ALT-NEXT: 	NewRootDepth: 16	RootDepth: 16	It MustReduceDepth 	  but it does NOT do it
; ALT-NEXT: add pattern FMA3
; ALT-NEXT: reassociating using pattern FMA3
; ALT-NEXT:   Dependence data for %21:vr64bit = {{.*}} WFMADB
; ALT-NEXT: 	NewRootDepth: 16	RootDepth: 22	It MustReduceDepth 	  and it does it
; ALT-NEXT: 		Resource length before replacement: 16 and after: 16
; ALT-NEXT: 		  As result it IMPROVES/PRESERVES Resource Length
; ALT-NEXT: add pattern FMA2_Add
; ALT-NEXT: reassociating using pattern FMA2_Add
; ALT-NEXT:   Dependence data for %23:vr64bit = {{.*}} WFMADB
; ALT-NEXT: 	NewRootDepth: 22	RootDepth: 28	It MustReduceDepth 	  and it does it
; ALT-NEXT: 		Resource length before replacement: 16 and after: 16
; ALT-NEXT: 		  As result it IMPROVES/PRESERVES Resource Length
; ALT-NEXT: add pattern FMA2_Add
; ALT-NEXT: reassociating using pattern FMA2_Add
; ALT-NEXT:   Dependence data for %25:vr64bit = {{.*}} WFMADB
; ALT-NEXT: 	NewRootDepth: 28	RootDepth: 34	It MustReduceDepth 	  and it does it
; ALT-NEXT: 		Resource length before replacement: 16 and after: 16
; ALT-NEXT: 		  As result it IMPROVES/PRESERVES Resource Length

; ALT:      # *** IR Dump After Machine InstCombiner (machine-combiner) ***:
; ALT:      %18:vr64bit = {{.*}} WFMADB killed %2:vr64bit, killed %3:vr64bit, %1:fp64bit
; ALT-NEXT: %29:vr64bit = {{.*}} WFMDB killed %4:vr64bit, killed %5:vr64bit
; ALT-NEXT: %30:vr64bit = {{.*}} WFMADB killed %6:vr64bit, killed %7:vr64bit, killed %18:vr64bit
; ALT-NEXT: %31:vr64bit = {{.*}} WFMADB killed %8:vr64bit, killed %9:vr64bit, %29:vr64bit
; ALT-NEXT: %32:vr64bit = {{.*}} WFMADB killed %10:vr64bit, killed %11:vr64bit, %30:vr64bit
; ALT-NEXT: %33:vr64bit = {{.*}} WFMADB killed %12:vr64bit, killed %13:vr64bit, %31:vr64bit
; ALT-NEXT: %34:vr64bit = {{.*}} WFMADB killed %14:vr64bit, killed %15:vr64bit, %32:vr64bit
; ALT-NEXT: %35:vr64bit = {{.*}} WFMADB killed %16:vr64bit, killed %17:vr64bit, %33:vr64bit
; ALT-NEXT: %25:vr64bit = {{.*}} WFADB_CCPseudo %34:vr64bit, %35:vr64bit

entry:
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 2
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 3
  %arrayidx6 = getelementptr inbounds double, ptr %x, i64 4
  %arrayidx8 = getelementptr inbounds double, ptr %x, i64 5
  %arrayidx10 = getelementptr inbounds double, ptr %x, i64 6
  %arrayidx12 = getelementptr inbounds double, ptr %x, i64 7
  %arrayidx14 = getelementptr inbounds double, ptr %x, i64 8
  %arrayidx16 = getelementptr inbounds double, ptr %x, i64 9
  %arrayidx18 = getelementptr inbounds double, ptr %x, i64 10
  %arrayidx20 = getelementptr inbounds double, ptr %x, i64 11
  %arrayidx22 = getelementptr inbounds double, ptr %x, i64 12
  %arrayidx24 = getelementptr inbounds double, ptr %x, i64 13
  %arrayidx26 = getelementptr inbounds double, ptr %x, i64 14
  %arrayidx28 = getelementptr inbounds double, ptr %x, i64 15

  %0 = load double, ptr %x
  %1 = load double, ptr %arrayidx1
  %2 = load double, ptr %arrayidx2
  %3 = load double, ptr %arrayidx4
  %4 = load double, ptr %arrayidx6
  %5 = load double, ptr %arrayidx8
  %6 = load double, ptr %arrayidx10
  %7 = load double, ptr %arrayidx12
  %8 = load double, ptr %arrayidx14
  %9 = load double, ptr %arrayidx16
  %10 = load double, ptr %arrayidx18
  %11 = load double, ptr %arrayidx20
  %12 = load double, ptr %arrayidx22
  %13 = load double, ptr %arrayidx24
  %14 = load double, ptr %arrayidx26
  %15 = load double, ptr %arrayidx28

  %mul1 = fmul reassoc nsz contract double %0, %1
  %mul2 = fmul reassoc nsz contract double %2, %3
  %mul3 = fmul reassoc nsz contract double %4, %5
  %mul4 = fmul reassoc nsz contract double %6, %7
  %mul5 = fmul reassoc nsz contract double %8, %9
  %mul6 = fmul reassoc nsz contract double %10, %11
  %mul7 = fmul reassoc nsz contract double %12, %13
  %mul8 = fmul reassoc nsz contract double %14, %15

  %A1 = fadd reassoc nsz contract double %A, %mul1
  %A2 = fadd reassoc nsz contract double %A1, %mul2
  %A3 = fadd reassoc nsz contract double %A2, %mul3
  %A4 = fadd reassoc nsz contract double %A3, %mul4
  %A5 = fadd reassoc nsz contract double %A4, %mul5
  %A6 = fadd reassoc nsz contract double %A5, %mul6
  %A7 = fadd reassoc nsz contract double %A6, %mul7
  %A8 = fadd reassoc nsz contract double %A7, %mul8

  ret double %A8
}

