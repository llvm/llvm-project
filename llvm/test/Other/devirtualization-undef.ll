; RUN: opt -passes='devirt<2>(function(simplifycfg))' %s -S | FileCheck %s

; CHECK: unreachable

declare void @llvm.assume(i1 noundef)
declare i1 @bar(ptr nonnull dereferenceable(1))

define void  @foo() {
  %a = call i1 null()
  call void @llvm.assume(i1 %a)
  ret void
}
