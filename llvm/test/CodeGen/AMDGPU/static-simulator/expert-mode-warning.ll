; RUN: llc -mtriple=amdgpu12.50 -amdgpu-enable-static-simulator %s -o /dev/null 2>&1 | FileCheck %s

; CHECK-NOT: warning: {{.*}}in function static_simulator_with_expert_scheduling_mode{{.*}}AMDGPU static simulator does not model non-expert scheduling mode, performance estimates may be overly optimistic
define amdgpu_kernel void @static_simulator_with_expert_scheduling_mode() #0 {
  ret void
}

; CHECK: warning: {{.*}}in function static_simulator_without_expert_scheduling_mode{{.*}}AMDGPU static simulator does not model non-expert scheduling mode, performance estimates may be overly optimistic
define amdgpu_kernel void @static_simulator_without_expert_scheduling_mode() #1 {
  ret void
}

attributes #0 = { "amdgpu-expert-scheduling-mode"="true" }
attributes #1 = { "amdgpu-expert-scheduling-mode"="false" }
