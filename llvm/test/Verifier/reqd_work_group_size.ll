; RUN: split-file %s %t
; RUN: llvm-as %t/valid64.ll --disable-output 2>&1 | count 0
; RUN: not llvm-as %t/overflow64.ll --disable-output 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: llvm-as %t/valid32.ll --disable-output 2>&1 | count 0
; RUN: not llvm-as %t/overflow32.ll --disable-output 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not llvm-as %t/operand-too-large32.ll --disable-output 2>&1 | FileCheck %s --check-prefix=OPERAND
; RUN: not llvm-as %t/operand-too-large64.ll --disable-output 2>&1 | FileCheck %s --check-prefix=OPERAND

; OVERFLOW: reqd_work_group_size product must fit in size_t
; OPERAND: reqd_work_group_size operands must fit in size_t

;--- valid64.ll
; UINT64_MAX * 1 * 1: product fits in 64-bit size_t.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @large_dim() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i64 -1, i32 1, i32 1}

;--- overflow64.ll
; UINT64_MAX * 2 * 1: product overflows 64-bit size_t.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @overflow_product() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i64 -1, i64 2, i32 1}

;--- valid32.ll
; UINT32_MAX * 1 * 1: product fits in 32-bit size_t.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv32-unknown-unknown"

define spir_kernel void @large_dim() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 -1, i32 1, i32 1}

;--- overflow32.ll
; UINT32_MAX * 2 * 1: product overflows 32-bit size_t.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv32-unknown-unknown"

define spir_kernel void @overflow_product() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 -1, i32 2, i32 1}

;--- operand-too-large32.ll
; i64 value > UINT32_MAX: operand does not fit in 32-bit size_t.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv32-unknown-unknown"

define spir_kernel void @operand_too_large() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i64 4294967296, i32 1, i32 1}

;--- operand-too-large64.ll
; i128 value > UINT64_MAX: operand does not fit in 64-bit size_t.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @operand_too_large() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i128 18446744073709551616, i32 1, i32 1}
