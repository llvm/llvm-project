;; Check that we infer the appropriate datalayout components for RVY pure-capability ABIs
; RUN: llc < %s -mtriple=riscv32 -target-abi=ilp32 --stop-after=pre-isel-intrinsic-lowering | FileCheck %s --check-prefix=CHECK-RVI32
; RUN: llc < %s -mtriple=riscv64 -target-abi=lp64 --stop-after=pre-isel-intrinsic-lowering | FileCheck %s --check-prefix=CHECK-RVI64
; RUN: llc < %s -mtriple=riscv32 -target-abi=il32pc64 --stop-after=pre-isel-intrinsic-lowering | FileCheck %s --check-prefix=CHECK-RVY32
; RUN: llc < %s -mtriple=riscv64 -target-abi=l64pc128 --stop-after=pre-isel-intrinsic-lowering | FileCheck %s --check-prefix=CHECK-RVY64

; RUN: opt -S < %s -mtriple=riscv32 -target-abi=ilp32 | FileCheck %s --check-prefix=CHECK-RVI32
; RUN: opt -S < %s -mtriple=riscv64 -target-abi=lp64 | FileCheck %s --check-prefix=CHECK-RVI64
; RUN: opt -S < %s -mtriple=riscv32 -target-abi=il32pc64 | FileCheck %s --check-prefix=CHECK-RVY32
; RUN: opt -S < %s -mtriple=riscv64 -target-abi=l64pc128 | FileCheck %s --check-prefix=CHECK-RVY64

; CHECK-RVI32: target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
; CHECK-RVI64: target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
; CHECK-RVY32: target datalayout = "e-m:e-p:32:32-pe200:64:64:64:32-i64:64-n32-S128-A200-P200-G200"
; CHECK-RVY64: target datalayout = "e-m:e-p:64:64-pe200:128:128:128:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"
