; RUN: not llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel-abort=2 -pass-remarks-missed="gisel.*" -o /dev/null 2>&1 %s | FileCheck -check-prefix=ERR %s

; ERR: remark: <unknown>:0:0: cannot select: %24:sreg_32(p5) = G_DYN_STACKALLOC %23:vgpr(s32), 1 (in function: kernel_dynamic_stackalloc_vgpr_align4)
; ERR-NEXT: warning: Instruction selection used fallback path for kernel_dynamic_stackalloc_vgpr_align4
; ERR-NEXT: error: <unknown>:0:0: in function kernel_dynamic_stackalloc_vgpr_align4 void (i32 addrspace(1)*): unsupported dynamic alloca

; ERR: remark: <unknown>:0:0: cannot select: %8:sreg_32(p5) = G_DYN_STACKALLOC %7:vgpr(s32), 1 (in function: func_dynamic_stackalloc_vgpr_align4)
; ERR-NEXT: warning: Instruction selection used fallback path for func_dynamic_stackalloc_vgpr_align4
; ERR-NEXT: error: <unknown>:0:0: in function func_dynamic_stackalloc_vgpr_align4 void (i32): unsupported dynamic alloca

define amdgpu_kernel void @kernel_dynamic_stackalloc_vgpr_align4(i32 addrspace(1)* %ptr) {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i32 %id
  %n = load i32, i32 addrspace(1)* %gep
  %alloca = alloca i32, i32 %n, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

define void @func_dynamic_stackalloc_vgpr_align4(i32 %n) {
  %alloca = alloca i32, i32 %n, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }
