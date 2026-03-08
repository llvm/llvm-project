; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto2 run -mattr=+zcmp -save-temps -filetype=asm -o %t.s %t.o -r=%t.o,func,p
; RUN: llvm-nm %t.o | FileCheck %s --check-prefix NM
; RUN: llvm-nm %t.s.0.5.precodegen.bc | FileCheck %s --check-prefix NM
; RUN: FileCheck %s --input-file %t.s.0

; NM: T func

; CHECK:      cm.mvsa01 s1, s0
; CHECK-NEXT: ret


target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64"

module asm ".globl func; func: cm.mvsa01 s1, s0; ret"

!llvm.module.flags = !{!0, !1, !2, !4, !7}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{i32 6, !"riscv-isa", !3}
!3 = !{!"rv64i2p1_c2p0_zca1p0_zcmp1p0"}
!4 = !{i32 6, !"global-asm-symbols", !5}
!5 = !{!6}
!6 = !{!"func", i32 2050}
!7 = !{i32 6, !"global-asm-symvers", !8}
!8 = !{}
