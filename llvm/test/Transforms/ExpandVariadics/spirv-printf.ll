; RUN: opt -mtriple=spirv64-unknown-unknown -S --passes=expand-variadics --expand-variadics-override=lowering %s | FileCheck %s
;
; An unmangled C `printf` declaration demangles to bare "printf" (no argument
; list). On SPIR-V it must be left as a variadic call so the backend can lower
; it to the OpenCL.std printf ExtInst with inline operands, instead of having
; its arguments packed into a vararg buffer here. A user-defined variadic
; function with a body is still expanded as usual.

@.str = private unnamed_addr addrspace(2) constant [4 x i8] c"%d\0A\00"

declare spir_func i32 @printf(ptr addrspace(2), ...)

; printf is left untouched: the call stays variadic and nothing is packed
; (CHECK-NEXT pins that ret immediately follows the call).
; CHECK-LABEL: define spir_kernel void @uses_printf(
; CHECK-NEXT:    {{%.*}} = call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i32 42)
; CHECK-NEXT:    ret void
define spir_kernel void @uses_printf() {
  %r = call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i32 42)
  ret void
}

; Control: an ordinary variadic function with a body is still expanded, i.e.
; its caller packs the arguments into a vararg buffer.
define spir_func i32 @my_variadic(...) {
  ret i32 0
}

; CHECK-LABEL: define spir_kernel void @uses_my_variadic(
; CHECK:    %vararg_buffer = alloca %uses_my_variadic.vararg
; CHECK:    call spir_func i32 @my_variadic(ptr %vararg_buffer)
define spir_kernel void @uses_my_variadic() {
  %r = call spir_func i32 (...) @my_variadic(i32 7)
  ret void
}
