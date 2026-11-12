; RUN: not opt -S -passes=kcfi %s 2>&1 | FileCheck %s

; CHECK: error: -fpatchable-function-entry=N,M, where M>0 is not compatible with -fsanitize=kcfi on this target
define void @f1(ptr noundef %x) #0 {
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-prefix"="1" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
