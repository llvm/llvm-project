; RUN: opt -disable-output -passes=objc-arc-contract < %s
; test that we don't crash on unreachable code
%2 = type opaque

define void @_i_Test__foo(ptr %x) {
entry:
  unreachable

return:                                           ; No predecessors!
  %foo = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  call void @callee()
  call void @use_pointer(ptr %foo)
  call void @llvm.objc.release(ptr %foo) nounwind
  ret void
}

declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare void @llvm.objc.release(ptr)
declare void @callee()
declare void @use_pointer(ptr)
