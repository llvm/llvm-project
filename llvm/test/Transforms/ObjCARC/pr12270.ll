; RUN: opt -disable-output -objc-arc-contract < %s
; test that we don't crash on unreachable code
%2 = type opaque

define void @_i_Test__foo(%2 *%x) {
entry:
  unreachable

return:                                           ; No predecessors!
  %bar = bitcast %2* %x to i8*
  %foo = call i8* @llvm.objc.autoreleaseReturnValue(i8* %bar) nounwind
  call void @callee()
  call void @use_pointer(i8* %foo)
  call void @llvm.objc.release(i8* %foo) nounwind
  ret void
}

declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare void @llvm.objc.release(i8*)
declare void @callee()
declare void @use_pointer(i8*)
