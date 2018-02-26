; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}if_with_kill:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]],
; GCN-NEXT: s_xor_b64 s[{{[0-9:]+}}], exec, [[SAVEEXEC]]
define amdgpu_ps void @if_with_kill(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  tail call void @llvm.AMDGPU.kilp()
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}if_with_loop_kill_after:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]],
; GCN-NEXT: s_xor_b64 s[{{[0-9:]+}}], exec, [[SAVEEXEC]]
define amdgpu_ps void @if_with_loop_kill_after(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  %sub = sub i32 %arg, 1
  br label %loop

loop:
  %ind = phi i32 [%sub, %then], [%dec, %loop]
  %dec = sub i32 %ind, 1
  %cc = icmp ne i32 %ind, 0
  br i1 %cc, label %loop, label %break

break:
  tail call void @llvm.AMDGPU.kilp()
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}if_with_kill_inside_loop:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]],
; GCN-NEXT: s_xor_b64 s[{{[0-9:]+}}], exec, [[SAVEEXEC]]
define amdgpu_ps void @if_with_kill_inside_loop(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  %sub = sub i32 %arg, 1
  br label %loop

loop:
  %ind = phi i32 [%sub, %then], [%dec, %loop]
  %dec = sub i32 %ind, 1
  %cc = icmp ne i32 %ind, 0
  tail call void @llvm.AMDGPU.kilp()
  br i1 %cc, label %loop, label %break

break:
  br label %endif

endif:
  ret void
}

declare void @llvm.AMDGPU.kilp()
