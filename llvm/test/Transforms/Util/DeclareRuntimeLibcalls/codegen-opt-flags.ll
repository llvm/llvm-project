; REQUIRES: arm-registered-target

; Make sure that codegen flags work to change the set of libcalls
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm-none-eabi -float-abi=hard -exception-model=sjlj < %s | FileCheck %s

; Depends on -exception-model
; CHECK: declare arm_aapcs_vfpcc void @_Unwind_SjLj_Register(ptr)
; CHECK: declare arm_aapcs_vfpcc void @_Unwind_SjLj_Resume(ptr)
; CHECK: declare arm_aapcs_vfpcc void @_Unwind_SjLj_Unregister(ptr)

; Calling convention depends on -float-abi
; CHECK: declare arm_aapcs_vfpcc fp128 @__addtf3(fp128, fp128)

; memclr functions depend on -meabi
; CHECK: declare arm_aapcscc void @__aeabi_memclr(ptr, i32)
; CHECK: declare arm_aapcscc void @__aeabi_memclr4(ptr, i32)
; CHECK: declare arm_aapcscc void @__aeabi_memclr8(ptr, i32)

; The AEABI memory functions do not return the destination.
; CHECK: declare arm_aapcscc void @__aeabi_memcpy(ptr noalias writeonly, ptr noalias readonly, i32) [[MEMCPY_ATTRS:#[0-9]+]]
; CHECK: declare arm_aapcscc void @__aeabi_memmove(ptr writeonly, ptr readonly, i32) [[MEMCPY_ATTRS]]
; CHECK: declare arm_aapcscc void @__aeabi_memset(ptr writeonly, i32, i32) [[MEMSET_ATTRS:#[0-9]+]]
; CHECK: declare arm_aapcs_vfpcc ptr @memcpy(ptr noalias returned writeonly, ptr noalias readonly, i32) [[MEMCPY_ATTRS]]

; CHECK: attributes [[MEMCPY_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
; CHECK: attributes [[MEMSET_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
