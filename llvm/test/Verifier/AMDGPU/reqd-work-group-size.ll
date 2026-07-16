; RUN: split-file %s %t

; RUN: not llvm-as %t/missing-flat.ll --disable-output 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not llvm-as %t/broad-flat.ll --disable-output 2>&1 | FileCheck %s --check-prefix=BROAD
; RUN: not llvm-as %t/wrong-flat.ll --disable-output 2>&1 | FileCheck %s --check-prefix=WRONG
; RUN: not llvm-as %t/malformed-flat.ll --disable-output 2>&1 | FileCheck %s --check-prefix=MALFORMED-FLAT
; RUN: not llvm-as %t/malformed-reqd.ll --disable-output 2>&1 | FileCheck %s --check-prefix=MALFORMED-REQD
; RUN: not llvm-as %t/non-integer-reqd.ll --disable-output 2>&1 | FileCheck %s --check-prefix=NON-INTEGER-REQD
; RUN: llvm-as %t/valid.ll --disable-output 2>&1 | count 0
; RUN: llvm-as %t/spir.ll --disable-output 2>&1 | count 0

; MISSING: reqd_work_group_size requires amdgpu-flat-work-group-size
; BROAD: amdgpu-flat-work-group-size must equal the product of reqd_work_group_size operands
; WRONG: amdgpu-flat-work-group-size must equal the product of reqd_work_group_size operands
; MALFORMED-FLAT: amdgpu-flat-work-group-size must be a pair of unsigned integers
; MALFORMED-REQD: reqd_work_group_size must have exactly three operands
; NON-INTEGER-REQD: reqd_work_group_size operands must be integer constants

;--- missing-flat.ll
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @missing_flat() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2, i32 1}

;--- broad-flat.ll
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @broad_flat() "amdgpu-flat-work-group-size"="16,128" !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2, i32 1}

;--- wrong-flat.ll
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @wrong_flat() "amdgpu-flat-work-group-size"="128,128" !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2, i32 1}

;--- malformed-flat.ll
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @malformed_flat() "amdgpu-flat-work-group-size"="64" !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2, i32 1}

;--- malformed-reqd.ll
target triple = "spirv64-unknown-unknown"

define spir_kernel void @malformed_reqd() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2}

;--- non-integer-reqd.ll
target triple = "spirv64-unknown-unknown"

define spir_kernel void @non_integer_reqd() !reqd_work_group_size !0 {
  ret void
}

!0 = !{!"32", i32 2, i32 1}

;--- valid.ll
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @valid() "amdgpu-flat-work-group-size"="64,64" !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2, i32 1}

;--- spir.ll
target triple = "spirv64-unknown-unknown"

define spir_kernel void @spir() !reqd_work_group_size !0 {
  ret void
}

!0 = !{i32 32, i32 2, i32 1}
