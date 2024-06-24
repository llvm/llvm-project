; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=DEFINE %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=DEFINE %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=DEFINE %s

; We have 4 function:
;   - Each function has an internal helper
;   - @A and @B's helpers does an indirect call.
;
; For non-kernels, indirect calls shouldn't matter, so
; @CallCandidate doesn't have to be in A/B's partition, unlike
; in the corresponding tests for kernels where it has to.

; CHECK0: define hidden void @HelperA
; CHECK0: define hidden void @HelperB
; CHECK0: define internal void @HelperC
; CHECK0: define internal void @HelperD
; CHECK0: define void @A
; CHECK0: define void @B

; CHECK1: define internal void @HelperD
; CHECK1: define void @D

; CHECK2: define hidden void @CallCandidate
; CHECK2: define internal void @HelperC
; CHECK2: define void @C

@addrthief = global [3 x ptr] [ptr @HelperA, ptr @HelperB, ptr @CallCandidate]

define internal void @HelperA(ptr %call) {
  call void %call()
  ret void
}

define internal void @HelperB(ptr %call) {
  call void @HelperC()
  call void %call()
  call void @HelperD()
  ret void
}

define internal void @CallCandidate() {
  ret void
}

define internal void @HelperC() {
  ret void
}

define internal void @HelperD() {
  ret void
}

define void @A(ptr %call) {
  call void @HelperA(ptr %call)
  ret void
}

define void @B(ptr %call) {
  call void @HelperB(ptr %call)
  ret void
}

define void @C() {
  call void @HelperC()
  ret void
}

define void @D() {
  call void @HelperD()
  ret void
}
