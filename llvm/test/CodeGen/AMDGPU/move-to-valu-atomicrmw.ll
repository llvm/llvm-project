; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; XUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; FIXME: broken on VI because flat instructions need to be emitted
; instead of addr64 equivalent of the _OFFSET variants.

; Check that moving the pointer out of the resource descriptor to
; vaddr works for atomics.

declare i32 @llvm.amdgcn.workitem.id.x() #1

; GCN-LABEL: {{^}}atomic_max_i32:
; GCN: buffer_atomic_smax v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:400 glc{{$}}
define void @atomic_max_i32(i32 addrspace(1)* %out, i32 addrspace(1)* addrspace(1)* %in, i32 addrspace(1)* %x, i32 %y) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.gep = getelementptr i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %in, i32 %tid
  %ptr = load volatile i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %tid.gep
  %xor = xor i32 %tid, 1
  %cmp = icmp ne i32 %xor, 0
  br i1 %cmp, label %atomic, label %exit

atomic:
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i32 100
  %ret = atomicrmw max i32 addrspace(1)* %gep, i32 %y seq_cst
  store i32 %ret, i32 addrspace(1)* %out
  br label %exit

exit:
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_noret:
; GCN: buffer_atomic_smax v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:400{{$}}
define void @atomic_max_i32_noret(i32 addrspace(1)* %out, i32 addrspace(1)* addrspace(1)* %in, i32 addrspace(1)* %x, i32 %y) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.gep = getelementptr i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %in, i32 %tid
  %ptr = load volatile i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %tid.gep
  %xor = xor i32 %tid, 1
  %cmp = icmp ne i32 %xor, 0
  br i1 %cmp, label %atomic, label %exit

atomic:
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i32 100
  %ret = atomicrmw max i32 addrspace(1)* %gep, i32 %y seq_cst
  br label %exit

exit:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
