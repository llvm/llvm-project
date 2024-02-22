; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/1.bc %s
; RUN: llvm-lto -print-macho-cpu-only %t/1.bc | FileCheck %s

target triple = "arm64e-apple-darwin"

!0 = !{ i32 5, i1 true }
!1 = !{ !0 }
!2 = !{ i32 6, !"ptrauth.abi-version", !1 }
!llvm.module.flags = !{ !2 }
; CHECK: 1.bc:
; CHECK-NEXT: cputype: 16777228
; CHECK-NEXT: cpusubtype: 3305111554
