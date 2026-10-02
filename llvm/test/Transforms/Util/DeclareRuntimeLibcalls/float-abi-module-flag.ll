; REQUIRES: arm-registered-target

; The "float-abi" module flag selects the calling convention of the
; runtime library calls.

; RUN: split-file %s %t

; Hard float ABI module flag: libcalls use the AAPCS-VFP calling convention.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm-none-linux-gnueabi < %t/hard.ll | FileCheck %s --check-prefix=HARD

; Soft float ABI module flag: libcalls use the plain AAPCS calling convention.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm-none-linux-gnueabi < %t/soft.ll | FileCheck %s --check-prefix=SOFT

; No module flag: the ABI defaults to the one implied by the target triple.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm-none-linux-gnueabihf < %t/none.ll | FileCheck %s --check-prefix=HARD
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm-none-linux-gnueabi < %t/none.ll | FileCheck %s --check-prefix=SOFT

;--- hard.ll
; HARD: declare arm_aapcs_vfpcc void @__addtf3(...)
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}

;--- soft.ll
; SOFT: declare arm_aapcscc void @__addtf3(...)
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}

;--- none.ll
