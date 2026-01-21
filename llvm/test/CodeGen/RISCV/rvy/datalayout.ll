;; Check that we infer the appropriate datalayout components for RVY (p200 for capabilities).
; RUN: llc < %s -mtriple=riscv32 -mattr=+experimental-y --stop-after=pre-isel-intrinsic-lowering | FileCheck %s --check-prefix=CHECK32
; RUN: llc < %s -mtriple=riscv64 -mattr=+experimental-y --stop-after=pre-isel-intrinsic-lowering | FileCheck %s --check-prefix=CHECK64
; RUN: opt -S < %s -mtriple=riscv32 -mattr=+experimental-y | FileCheck %s --check-prefix=CHECK32
; RUN: opt -S < %s -mtriple=riscv64 -mattr=+experimental-y | FileCheck %s --check-prefix=CHECK64

; CHECK32: target datalayout = "e-m:e-p:32:32-pe200:64:64:64:32-i64:64-n32-S128"
; CHECK64: target datalayout = "e-m:e-p:64:64-pe200:128:128:128:64-i64:64-i128:128-n32:64-S128"
