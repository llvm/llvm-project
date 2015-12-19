; RUN: not llc -O0 -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: unsupported addrspacecast not implemented

; XUN: llc -O0 -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-NO-PROMOTE %s
; XUN: llc -O0 -march=amdgcn -mcpu=bonaire -mattr=+promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-PROMOTE %s
; XUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=-promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-NO-PROMOTE %s
; XUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=+promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-PROMOTE %s

; Disable optimizations in case there are optimizations added that
; specialize away generic pointer accesses.

; CHECK-LABEL: {{^}}branch_use_flat_i32:
; CHECK: flat_store_dword {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}
; CHECK: s_endpgm
define void @branch_use_flat_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* %gptr, i32 addrspace(3)* %lptr, i32 %x, i32 %c) #0 {
entry:
  %cmp = icmp ne i32 %c, 0
  br i1 %cmp, label %local, label %global

local:
  %flat_local = addrspacecast i32 addrspace(3)* %lptr to i32 addrspace(4)*
  br label %end

global:
  %flat_global = addrspacecast i32 addrspace(1)* %gptr to i32 addrspace(4)*
  br label %end

end:
  %fptr = phi i32 addrspace(4)* [ %flat_local, %local ], [ %flat_global, %global ]
  store i32 %x, i32 addrspace(4)* %fptr, align 4
;  %val = load i32, i32 addrspace(4)* %fptr, align 4
;  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; TODO: This should not be zero when registers are used for small
; scratch allocations again.

; Check for prologue initializing special SGPRs pointing to scratch.
; CHECK-LABEL: {{^}}store_flat_scratch:
; CHECK: s_movk_i32 flat_scratch_lo, 0
; CHECK-NO-PROMOTE: s_movk_i32 flat_scratch_hi, 0x28{{$}}
; CHECK-PROMOTE: s_movk_i32 flat_scratch_hi, 0x0{{$}}
; CHECK: flat_store_dword
; CHECK: s_barrier
; CHECK: flat_load_dword
define void @store_flat_scratch(i32 addrspace(1)* noalias %out, i32) #0 {
  %alloca = alloca i32, i32 9, align 4
  %x = call i32 @llvm.r600.read.tidig.x() #3
  %pptr = getelementptr i32, i32* %alloca, i32 %x
  %fptr = addrspacecast i32* %pptr to i32 addrspace(4)*
  store i32 %x, i32 addrspace(4)* %fptr
  ; Dummy call
  call void @llvm.AMDGPU.barrier.local() #1
  %reload = load i32, i32 addrspace(4)* %fptr, align 4
  store i32 %reload, i32 addrspace(1)* %out, align 4
  ret void
}

declare void @llvm.AMDGPU.barrier.local() #1
declare i32 @llvm.r600.read.tidig.x() #3

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #3 = { nounwind readnone }
