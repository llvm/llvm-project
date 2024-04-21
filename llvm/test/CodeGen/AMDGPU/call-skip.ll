; XFAIL: *
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; A call should be skipped if all lanes are zero, since we don't know
; what side effects should be avoided inside the call.
define hidden void @func() #1 {
  ret void
}

; GCN-LABEL: {{^}}if_call:
; GCN: s_and_saveexec_b64
; GCN-NEXT: s_cbranch_execz [[END:.LBB[0-9]+_[0-9]+]]
; GCN: s_swappc_b64
; GCN: [[END]]:
define void @if_call(i32 %flag) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %call, label %end

call:
  call void @func()
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}if_asm:
; GCN: s_and_saveexec_b64
; GCN-NEXT: s_cbranch_execz [[END:.LBB[0-9]+_[0-9]+]]
; GCN: ; sample asm
; GCN: [[END]]:
define void @if_asm(i32 %flag) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %call, label %end

call:
  call void asm sideeffect "; sample asm", ""()
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}if_call_kernel:
; GCN: s_and_saveexec_b64
; GCN-NEXT: s_cbranch_execz .LBB3_2
; GCN: s_swappc_b64
define amdgpu_kernel void @if_call_kernel() #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cc = icmp eq i32 %id, 0
  br i1 %cc, label %call, label %end

call:
  call void @func()
  br label %end

end:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind noinline }
attributes #2 = { nounwind readnone speculatable }
