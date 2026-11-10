; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -global-isel -mtriple=amdgpu9.50-amd-amdhsa -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -global-isel -mtriple=amdgpu9.50-amd-amdhsa %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -enable-new-pm -mtriple=amdgpu9.50-amd-amdhsa %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa %s -o - | FileCheck %s --check-prefix=NO-PAD --implicit-check-not=buffer_inv

; Both instruction selectors keep the hint in the arm that requests padding.
; ISEL-LABEL: name: hint_counter_intrinsic
; ISEL: bb.{{[0-9]+}}.long:
; ISEL: GLOBAL_LOAD_DWORD
; ISEL-NEXT: PAD_WAITCNT 0, 1
; ISEL: bb.{{[0-9]+}}.join:

; The padding belongs to the skipped edge, not beside the extra load, so the
; store of %a only waits for the older of two outstanding loads.
; ASM-LABEL: hint_counter_intrinsic:
; ASM: global_load_dword
; ASM: s_cbranch_scc1 [[PAD:\.LBB[0-9]+_[0-9]+]]
; ASM: global_load_dword
; ASM-NOT: buffer_inv
; ASM: [[JOIN:\.LBB[0-9]+_[0-9]+]]:
; ASM-NEXT: s_waitcnt vmcnt(1)
; ASM-NEXT: global_store_dword
; ASM: s_endpgm
; ASM: [[PAD]]:
; ASM-NEXT: buffer_inv
; ASM-NEXT: s_branch [[JOIN]]

; Without BUFFER_INV, the hint is ignored and the store waits for both loads.
; NO-PAD-LABEL: hint_counter_intrinsic:
; NO-PAD: ; %long
; NO-PAD: global_load_dword
; NO-PAD-NEXT: ; pad_waitcnt(0, 1)
; NO-PAD: s_waitcnt vmcnt(0)
; NO-PAD-NEXT: global_store_dword
define amdgpu_kernel void @hint_counter_intrinsic(ptr addrspace(1) %data,
                                                  ptr addrspace(1) %out,
                                                  i32 %condition) {
entry:
  %index = call i32 @llvm.amdgcn.workitem.id.x()
  %wide = zext i32 %index to i64
  %a.ptr = getelementptr i32, ptr addrspace(1) %data, i64 %wide
  %a = load i32, ptr addrspace(1) %a.ptr, align 4
  %offset = add i64 %wide, 64
  %cond = icmp ne i32 %condition, 0
  br i1 %cond, label %long, label %join
long:
  %b.ptr = getelementptr i32, ptr addrspace(1) %data, i64 %offset
  %b = load i32, ptr addrspace(1) %b.ptr, align 4
  call void @llvm.amdgcn.pad.waitcnt(i32 0, i32 1)
  br label %join
join:
  %v = phi i32 [ %b, %long ], [ 0, %entry ]
  %a.out = getelementptr i32, ptr addrspace(1) %out, i64 %wide
  store i32 %a, ptr addrspace(1) %a.out, align 4
  %v.out = getelementptr i32, ptr addrspace(1) %out, i64 %offset
  store i32 %v, ptr addrspace(1) %v.out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.pad.waitcnt(i32 immarg, i32 immarg)
