; RUN: opt -S -passes=spirv-remove-aggregate-types -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

@fp = global ptr addrspace(4) @callback
; CHECK: @fp = global ptr addrspace(4) @callback

; @fp keeps pointing at @callback across the clone/rename via RAUW.
; CHECK-LABEL: define void @caller(
; CHECK: %ptr = load ptr addrspace(4), ptr @fp
; CHECK: call addrspace(4) i32 %ptr(i32 0), !spv.mutated_callsite ![[#MUTATED_CS:]]
define void @caller() {
  %ptr = load ptr addrspace(4), ptr @fp
  %r = call addrspace(4) { float, float } %ptr(i32 0)
  ret void
}

; CHECK-LABEL: define i32 @callback(
define { float, float } @callback({ float, float } %x) addrspace(4) {
  ret { float, float } %x
}

; nofpclass is dropped from the i32 placeholders and kept elsewhere.
; CHECK-LABEL: define void @nofpclass_caller(
; CHECK: call noundef i32 @nofpclass_callee(i32 noundef %x, float nofpclass(nan) %f)
; CHECK: call i32 %fp(i32 %x), !spv.mutated_callsite
define void @nofpclass_caller(ptr %fp, [2 x float] %x, float %f) {
  %r = call noundef nofpclass(nan) [2 x float] @nofpclass_callee([2 x float] noundef nofpclass(nan) %x, float nofpclass(nan) %f)
  %s = call nofpclass(nan) [2 x float] %fp([2 x float] nofpclass(nan) %x)
  ret void
}

; CHECK-LABEL: define noundef i32 @nofpclass_callee(i32 noundef %a, float nofpclass(nan) %f)
define noundef nofpclass(nan) [2 x float] @nofpclass_callee([2 x float] noundef nofpclass(nan) %a, float nofpclass(nan) %f) {
  ret [2 x float] %a
}

; CHECK-DAG: ![[#MUTATED_CS]] = !{!"spv.mutated_callsite.caller.0"}
; CHECK-DAG: !spv.cloned_funcs = !{![[#CLONED:]], ![[#]], ![[#]]}
; CHECK-DAG: ![[#CLONED]] = !{!"callback", ![[#RET:]], ![[#ARG:]]}
; CHECK-DAG: ![[#RET]] = !{i32 -1, { float, float } zeroinitializer}
; CHECK-DAG: ![[#ARG]] = !{i32 0, { float, float } zeroinitializer}
