; RUN: llc -mtriple=amdgcn -mcpu=tahiti -mattr=-enable-ds128 < %s | FileCheck -check-prefixes=SI,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-enable-ds128 < %s | FileCheck -check-prefixes=CIVI,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-enable-ds128 < %s | FileCheck -check-prefixes=CIVI,GCN %s

; Test if ds_read/write_b128 doesn't gets generated when the option is
; disabled.
; GCN-LABEL: {{^}}local_v4f32_to_2b64
;
; SI-NOT: ds_read_b128
; SI-NOT: ds_write_b128
;
; CIVI: ds_read2_b64
; CIVI: ds_write2_b64
define amdgpu_kernel void @local_v4f32_to_2b64(ptr addrspace(3) %out, ptr addrspace(3) %in) {
  %ld = load <4 x float>, ptr addrspace(3) %in, align 16
  store <4 x float> %ld, ptr addrspace(3) %out, align 16
  ret void
}

attributes #0 = { nounwind }
