; REQUIRES: asserts

; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -stop-after=postmisched -debug-only=machine-scheduler 2>&1 >/dev/null | FileCheck %s

define void @pr47155() {
; CHECK: *** Final schedule for %bb.0 ***
; CHECK: ********** MI Scheduling **********
; CHECK-NEXT: pr47155:%bb.0 entry
; CHECK:      SU(0):   INLINEASM &"mtlr 31"{{.*}}implicit-def early-clobber $lr
; CHECK:      Successors:
; CHECK-NEXT:   SU(1): Out  Latency=0
; CHECK-NEXT:   SU(1): Ord  Latency=0 Barrier
; CHECK-NEXT: SU(1):   INLINEASM &"mtlr 31"{{.*}}implicit-def early-clobber $lr8
; CHECK:      Predecessors:
; CHECK-NEXT:   SU(0): Out  Latency=0
; CHECK-NEXT:   SU(0): Ord  Latency=0 Barrier
; CHECK-NEXT: ExitSU:
entry:
  call void asm sideeffect "mtlr 31", "~{lr}"()
  call void asm sideeffect "mtlr 31", "~{lr8}"()
  ret void
}

define void @pr47156(ptr %fn) {
; CHECK: *** Final schedule for %bb.0 ***
; CHECK: ********** MI Scheduling **********
; CHECK-NEXT: pr47156:%bb.0 entry
; CHECK:      SU(0):   INLINEASM &"mtctr 31"{{.*}}implicit-def early-clobber $ctr
; CHECK:      Successors:
; CHECK-NEXT:   SU(1): Out  Latency=0
; CHECK-NEXT: SU(1):   MTCTR8 renamable $x3, implicit-def $ctr8
; CHECK:      Predecessors:
; CHECK-NEXT:   SU(0): Out  Latency=0
; CHECK-NEXT: Successors:
; CHECK-NEXT:  ExitSU:
; CHECK-NEXT: SU(2):
entry:
  call void asm sideeffect "mtctr 31", "~{ctr}"()
  tail call void %fn()
  ret void
}

