; RUN: mlir-translate -import-llvm %s | FileCheck %s

declare void @callee(i32)
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: llvm.func @direct(
; CHECK-SAME: %[[PTR:.*]]: !llvm.ptr, %[[ARG:.*]]: i32)
; CHECK: %[[SEVEN:.*]] = llvm.mlir.constant(7 : i32) : i32
; CHECK: llvm.call @callee(%[[ARG]]) ["tag"(%[[SEVEN]], %[[PTR]] : i32, !llvm.ptr), "empty"(), "tag"(%[[ARG]] : i32)] : (i32 {llvm.noundef}) -> ()
define void @direct(ptr %p, i32 %arg) {
  call void @callee(i32 noundef %arg) ["tag"(i32 7, ptr %p), "empty"(), "tag"(i32 %arg)]
  ret void
}

; CHECK-LABEL: llvm.func @indirect_variadic(
; CHECK-SAME: %[[FN:.*]]: !llvm.ptr, %[[ARG:.*]]: i32, %[[PTR:.*]]: !llvm.ptr)
; CHECK: %[[RESULT:.*]] = llvm.call %[[FN]](%[[ARG]], %[[PTR]]) vararg(!llvm.func<i32 (i32, ...)>) ["deopt"(%[[ARG]] : i32)] : !llvm.ptr, (i32 {llvm.noundef}, !llvm.ptr) -> (i32 {llvm.signext})
; CHECK: llvm.return %[[RESULT]] : i32
define i32 @indirect_variadic(ptr %fn, i32 %arg, ptr %p) {
  %result = call signext i32 (i32, ...) %fn(i32 noundef %arg, ptr %p) ["deopt"(i32 %arg)]
  ret i32 %result
}

; CHECK-LABEL: llvm.func @shared_metadata(
; CHECK-SAME: %[[ARG:.*]]: i32, %[[COND:.*]]: i1)
; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"payload">
; CHECK: llvm.cond_br %[[COND]]
; CHECK: llvm.call @callee(%[[ARG]]) ["tag"(%[[MD]] : !llvm.metadata)] : (i32) -> ()
; CHECK: llvm.call @callee(%[[ARG]]) ["tag"(%[[MD]] : !llvm.metadata)] : (i32) -> ()
define void @shared_metadata(i32 %arg, i1 %cond) {
  br i1 %cond, label %left, label %right
left:
  call void @callee(i32 %arg) ["tag"(metadata !"payload")]
  ret void
right:
  call void @callee(i32 %arg) ["tag"(metadata !"payload")]
  ret void
}

; CHECK-LABEL: llvm.func @invoke_with_phis(
; CHECK-SAME: %[[FN:.*]]: !llvm.ptr, %[[ARG:.*]]: i32)
; CHECK: %[[RESULT:.*]] = llvm.invoke %[[FN]](%[[ARG]]) to ^[[NORMAL:bb[0-9]+]](%[[ARG]] : i32) unwind ^[[UNWIND:bb[0-9]+]](%[[ARG]] : i32) ["tag"(%[[FN]] : !llvm.ptr), "empty"()] : !llvm.ptr, (i32 {llvm.noundef}) -> (i32 {llvm.signext})
; CHECK: ^[[NORMAL]](%[[NORMAL_ARG:.*]]: i32):
; CHECK: llvm.add %[[RESULT]], %[[NORMAL_ARG]] : i32
; CHECK: ^[[UNWIND]](%[[UNWIND_ARG:.*]]: i32):
; CHECK: llvm.return %[[UNWIND_ARG]] : i32
define i32 @invoke_with_phis(ptr %fn, i32 %arg) personality ptr @__gxx_personality_v0 {
  %result = invoke signext i32 %fn(i32 noundef %arg) ["tag"(ptr %fn), "empty"()]
      to label %normal unwind label %unwind

normal:
  %normal_arg = phi i32 [%arg, %0]
  %sum = add i32 %result, %normal_arg
  ret i32 %sum

unwind:
  %unwind_arg = phi i32 [%arg, %0]
  %exception = landingpad {ptr, i32} cleanup
  ret i32 %unwind_arg
}
