; REQUIRES: arm-registered-target

; Make sure that codegen flags work to change the set of libcalls
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm-none-linux-gnueabi -float-abi=hard -exception-model=sjlj -meabi=4 < %s | FileCheck %s

; Depends on -exception-model
; CHECK: declare arm_aapcs_vfpcc void @_Unwind_SjLj_Register(...)
; CHECK: declare arm_aapcs_vfpcc void @_Unwind_SjLj_Resume(...)
; CHECK: declare arm_aapcs_vfpcc void @_Unwind_SjLj_Unregister(...)

; Calling convention depends on -float-abi
; CHECK: declare arm_aapcs_vfpcc void @__addtf3(...)

; memclr functions depend on -meabi
; CHECK: declare arm_aapcscc void @__aeabi_memclr(...)
; CHECK: declare arm_aapcscc void @__aeabi_memclr4(...)
; CHECK: declare arm_aapcscc void @__aeabi_memclr8(...)
