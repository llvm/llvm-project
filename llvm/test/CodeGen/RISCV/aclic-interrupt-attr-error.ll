; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 %t/m-preemptible.ll -o - 2>&1 \
; RUN:   | FileCheck %s --check-prefix=M-PREEMPTIBLE
; RUN: not llc -mtriple=riscv32 -mattr=+experimental-smnip \
; RUN:   %t/m-push-pop.ll -o - 2>&1 | FileCheck %s --check-prefix=M-PUSH-POP
; RUN: not llc -mtriple=riscv32 -mattr=+experimental-smnip \
; RUN:   %t/s-preemptible.ll -o - 2>&1 \
; RUN:   | FileCheck %s --check-prefix=S-PREEMPTIBLE
; RUN: not llc -mtriple=riscv32 -mattr=+experimental-smip \
; RUN:   %t/s-push-pop.ll -o - 2>&1 | FileCheck %s --check-prefix=S-PUSH-POP

; M-PREEMPTIBLE: 'preemptible' interrupt kinds require Smnip/Ssnip extension
; M-PUSH-POP: 'push-pop' interrupt kinds require Smip/Ssip extension
; S-PREEMPTIBLE: 'preemptible' interrupt kinds require Smnip/Ssnip extension
; S-PUSH-POP: 'push-pop' interrupt kinds require Smip/Ssip extension

;--- m-preemptible.ll
define void @f() "interrupt"="machine-preemptible" {
  ret void
}

;--- m-push-pop.ll
define void @f() "interrupt"="machine-preemptible-push-pop" {
  ret void
}

;--- s-preemptible.ll
define void @f() "interrupt"="supervisor-preemptible" {
  ret void
}

;--- s-push-pop.ll
define void @f() "interrupt"="supervisor-push-pop" {
  ret void
}
