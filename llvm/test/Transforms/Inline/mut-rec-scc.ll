; RUN: opt -S -passes='cgscc(inline,instcombine)' < %s | FileCheck %s
; RUN: opt -S -intra-scc-cost-multiplier=3 -passes='cgscc(inline,instcombine)' < %s | FileCheck %s --check-prefix=THREE

; We use call to a dummy function to avoid inlining test1 into test2 or vice
; versa, such that we aren't left with a trivial cycle, as trivial cycles are
; special-cased to never be inlined.
; However, InstCombine will eliminate these calls after inlining, and thus
; make the functions eligible for inlining in their callers.
declare void @dummy() readnone nounwind willreturn

define void @test1() {
; CHECK-LABEL: define void @test1(
; CHECK-NEXT:    call void @test2()
; CHECK-NEXT:    call void @test2()
; CHECK-NEXT:    ret void
;
  call void @test2()
  call void @test2()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  ret void
}

define void @test2() {
; CHECK-LABEL: define void @test2(
; CHECK-NEXT:    call void @test1()
; CHECK-NEXT:    call void @test1()
; CHECK-NEXT:    ret void
;
  call void @test1()
  call void @test1()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  ret void
}

; The inlined call sites should have the "function-inline-cost-multiplier" call site attribute.
; This test is a bit fragile in the exact number of inlining that happens based on thresholds.
define void @test3() {
; CHECK-LABEL: define void @test3(
; CHECK-NEXT:    call void @test2() #[[COSTMULT:[0-9]+]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    call void @test2() #[[COSTMULT]]
; CHECK-NEXT:    ret void
;
  call void @test2()
  call void @test2()
  ret void
}

; CHECK: [[COSTMULT]] = { "function-inline-cost-multiplier"="4" }
; THREE: "function-inline-cost-multiplier"="9"
