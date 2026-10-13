define void @input_off() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 0}
