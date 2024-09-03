; RUN: llc -mtriple=x86_64-pc-windows-msvc19.41.34120 %s

define void @foo() personality ptr @__CxxFrameHandler3 {
  call void @llvm.seh.scope.begin()
  unreachable
}

declare i32 @__CxxFrameHandler3(...)

declare void @llvm.seh.scope.begin()

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
