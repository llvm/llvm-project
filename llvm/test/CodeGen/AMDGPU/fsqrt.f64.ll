; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}v_safe_fsqrt_f64:
; GCN: v_sqrt_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define void @v_safe_fsqrt_f64(double addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %r0 = load double, double addrspace(1)* %in
  %r1 = call double @llvm.sqrt.f64(double %r0)
  store double %r1, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_unsafe_fsqrt_f64:
; GCN: v_sqrt_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define void @v_unsafe_fsqrt_f64(double addrspace(1)* %out, double addrspace(1)* %in) #2 {
  %r0 = load double, double addrspace(1)* %in
  %r1 = call double @llvm.sqrt.f64(double %r0)
  store double %r1, double addrspace(1)* %out
  ret void
}

declare double @llvm.sqrt.f64(double %Val) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "unsafe-fp-math"="false" }
attributes #2 = { nounwind "unsafe-fp-math"="true" }
