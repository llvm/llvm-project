target triple = "arm64-apple-ios7.0"

; RUN: llvm-dis < %S/upgrade-arc-runtime-calls-bitcast.bc | FileCheck %s

; CHECK: tail call i8* @objc_retain(i32 1)
; CHECK: tail call i8* @objc_storeStrong(

define void @testRuntimeCalls(i8* %a, i8** %b) {
  %v6 = tail call i8* @objc_retain(i32 1)
  %1 = tail call i8* @objc_storeStrong(i8** %b, i8* %a)
  ret void
}

declare i8* @objc_retain(i32)
declare i8* @objc_storeStrong(i8**, i8*)

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09fp, fp\09\09; marker for objc_retainAutoreleaseReturnValue"}
