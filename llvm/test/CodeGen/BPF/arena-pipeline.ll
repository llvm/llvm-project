; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack,+arena-data \
; RUN:   -debug-pass=Structure -o /dev/null < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -debug-pass=Structure -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck --check-prefix=NONE %s

; Locals are picked before pointers are cast.

; CHECK:    BPF Arena Data
; CHECK:    BPF Arena Kptr
; CHECK:    BPF Arena Stack
; CHECK:    BPF Arena Cast
; NONE-NOT: BPF Arena

define void @f() {
  ret void
}
