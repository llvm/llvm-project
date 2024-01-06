; RUN: llc < %s -relocation-model=pic -mtriple=sparc | FileCheck --check-prefix=CALL %s
; RUN: llc < %s -relocation-model=pic -mtriple=sparcv9 -mcpu=ultrasparc | FileCheck --check-prefix=CALL %s
; RUN: llc < %s -relocation-model=pic -mtriple=sparcv9 | FileCheck --check-prefix=RDPC %s

;; SPARC32 and SPARC64 for classic UltraSPARCs implement GETPCX
;; with a fake `call`.
;; All other SPARC64 targets implement it with `rd %pc, %o7`.

@value = external global i32

; CALL: call
; CALL-NOT: rd %pc
; RDPC: rd %pc
; RDPC-not: call
define i32 @test() {
  %1 = load i32, i32* @value
  ret i32 %1
}
