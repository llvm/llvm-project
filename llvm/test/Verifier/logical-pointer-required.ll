; RUN: llvm-as -disable-output %s

%S = type { i32, i32 }

define void @structured_gep_allowed(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([0 x %S]) %src, i32 %index, i32 1)
  ret void
}

define void @structured_alloca_allowed(ptr %src, i32 %index) {
entry:
  %tmp = call elementtype(i32) ptr @llvm.structured.alloca()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"require-logical-pointer", i32 1}
