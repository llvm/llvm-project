; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; In moveToVALU(), move to vector ALU is performed, all instrs in
; the use chain will be visited. We do not want the same node to be 
; pushed to the visit worklist more than once.
		
; GCN-LABEL: {{^}}in_worklist_once:
; GCN: buffer_load_dword
; GCN: BB0_1:
; GCN: v_xor_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: v_xor_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_and_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: v_and_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @in_worklist_once() #0 {
bb:
	%tmp = load i64, ptr addrspace(5) undef
br label %bb1

bb1:                                              ; preds = %bb1, %bb
	%tmp2 = phi i64 [ undef, %bb ], [ %tmp16, %bb1 ]
	%tmp3 = phi i64 [ %tmp, %bb ], [ undef, %bb1 ]
	%tmp11 = shl i64 %tmp2, 14
	%tmp13 = xor i64 %tmp11, %tmp2
	%tmp15 = and i64 %tmp3, %tmp13
	%tmp16 = xor i64 %tmp15, %tmp3
br label %bb1
}

attributes #0 = { nounwind }
