; RUN: opt -mtriple=spirv64-unknown-unknown -S --passes=expand-variadics --expand-variadics-override=lowering %s | FileCheck %s

; An indirect variadic call has no called Function. The SPIR-V ABI's
; ignoreFunction() override dereferences its argument unconditionally, so the
; pass must not query it for indirect calls. Regression test for a null
; pointer dereference in expandCall().

; CHECK-LABEL: define void @caller(
; CHECK: %vararg_buffer = alloca %caller.vararg
; CHECK: call void %fp(i32 %x, ptr %vararg_buffer)
; CHECK: ret void
define void @caller(ptr %fp, i32 %x) {
entry:
  call void (i32, ...) %fp(i32 %x)
  ret void
}
