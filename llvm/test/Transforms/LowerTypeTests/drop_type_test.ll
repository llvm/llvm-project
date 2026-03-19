; RUN: opt -S -passes=lowertypetests -lowertypetests-drop-type-tests=all < %s | FileCheck %s

define void @func() {
entry:
  %0 = tail call i1 @llvm.type.test(ptr null, metadata !"foo")
  br i1 %0, label %exit, label %trap

trap:
  unreachable

exit:
  ret void
  ; CHECK-LABEL: entry:
  ;  CHECK-NEXT: br i1 true, label %exit, label %trap
  ; CHECK-LABEL: trap:
  ;  CHECK-NEXT: unreachable
  ; CHECK-LABEL: exit:
  ;  CHECK-NEXT: ret void
}

declare i1 @llvm.type.test(ptr, metadata) #0
attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
