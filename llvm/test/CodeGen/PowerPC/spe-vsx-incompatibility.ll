; Adding -enable-matrix, which is disabled by default, forces the initialization
; of the PPCSubtarget which verifies the incompatible CPU features.
; RUN: not llc -mtriple=powerpcspe -mattr=+vsx -enable-matrix < %s 2>&1  | FileCheck %s

; CHECK: SPE and traditional floating point cannot both be enabled
define void @test() {
    ret void
}
