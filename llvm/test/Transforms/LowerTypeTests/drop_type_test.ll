; RUN: opt -S -passes=lowertypetests -lowertypetests-force-drop-type-tests -lowertypetests-drop-type-tests < %s | FileCheck %s

define void @func() {
entry:
  %0 = tail call i1 @llvm.type.test(ptr null, metadata !"foo")
  br i1 %0, label %exit, label %trap
  ;      CHECK: entry:
  ; CHECK-NEXT: br i1 true, label %exit, label %trap
  ; CHECK-NOT: @llvm.type.test

trap:
  unreachable

exit:
  ret void
}

declare i1 @llvm.type.test(ptr, metadata) #0
attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
