; RUN: llc -mtriple=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; This used to raise an assertion due to how the choice between uniform and
; non-uniform branches was determined.
;
; CHECK-LABEL: {{^}}main:
; CHECK: s_cbranch_vccnz
define amdgpu_ps float @main(ptr addrspace(8) inreg %rsrc) {
main_body:
  %v = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 1)
  %cc = fcmp une float %v, 1.000000e+00
  br i1 %cc, label %if, label %else

if:
  %u = fadd float %v, %v
  call void asm sideeffect "", ""() #0 ; Prevent ifconversion
  br label %else

else:
  %r = phi float [ %v, %main_body ], [ %u, %if ]
  ret float %r
}

declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32 immarg) #0

attributes #0 = { nounwind memory(argmem: read) }
