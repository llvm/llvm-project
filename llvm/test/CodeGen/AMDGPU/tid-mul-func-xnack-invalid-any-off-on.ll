; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s 2>&1 | FileCheck --check-prefixes=ERR %s

; ERR: error: xnack setting of 'func2' function does not match module xnack setting

define void @func0() {
entry:
  ret void
}

define void @func1() #0 {
entry:
  ret void
}

define void @func2() #1 {
entry:
  ret void
}

attributes #0 = { "target-features"="-xnack" }
attributes #1 = { "target-features"="+xnack" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
