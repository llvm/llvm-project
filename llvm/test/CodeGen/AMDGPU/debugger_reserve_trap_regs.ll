; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=+amdgpu-debugger-reserve-trap-regs -verify-machineinstrs < %s | FileCheck %s

; CHECK: reserved_vgpr_count = 4
; CHECK: ReservedVGPRCount: 4

; Function Attrs: nounwind
define void @debugger_reserve_trap_regs(i32 addrspace(1)* %A) #0 {
entry:
  %A.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %A, i32 addrspace(1)** %A.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i32 0
  store i32 1, i32 addrspace(1)* %arrayidx, align 4
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %1, i32 1
  store i32 2, i32 addrspace(1)* %arrayidx1, align 4
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %2, i32 2
  store i32 3, i32 addrspace(1)* %arrayidx2, align 4
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %3, i32 4
  store i32 4, i32 addrspace(1)* %arrayidx3, align 4
  ret void
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="fiji" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.kernels = !{!0}
!llvm.ident = !{!6}

!0 = !{void (i32 addrspace(1)*)* @debugger_reserve_trap_regs, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 1}
!2 = !{!"kernel_arg_access_qual", !"none"}
!3 = !{!"kernel_arg_type", !"int*"}
!4 = !{!"kernel_arg_base_type", !"int*"}
!5 = !{!"kernel_arg_type_qual", !""}
!6 = !{!"clang version 3.9.0 (trunk 266639)"}
