; RUN: llvm-split -enable-split-module-CG=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s

; CHECK0-DAG: define hidden void @__llvmsplit_unnamed()

define internal void @0() {
  ret void
}