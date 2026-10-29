; RUN: llc < %s -mtriple=wasm32-unknown-unknown -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+tail-call | FileCheck -DPTR=i32 %s
; RUN: llc < %s -mtriple=wasm64-unknown-unknown -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+tail-call | FileCheck -DPTR=i64 %s

; Test swifttailcc (SwiftTail calling convention) support for WebAssembly.
; Requires the tail-call feature for return_call / return_call_indirect.

; Basic swifttailcc function definition with swiftasync parameter.
; Missing swiftself and swifterror are padded automatically.
define swifttailcc void @basic_swifttailcc(ptr swiftasync %ctx) {
; CHECK-LABEL: basic_swifttailcc:
; CHECK:         .functype basic_swifttailcc ([[PTR]], [[PTR]], [[PTR]]) -> ()
  ret void
}

; All Swift parameter attributes together — no padding needed.
define swifttailcc void @full_swift_params(i32 %x, ptr swiftasync %ctx, ptr swiftself %self, ptr swifterror %err) {
; CHECK-LABEL: full_swift_params:
; CHECK:         .functype full_swift_params (i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  ret void
}

; Direct musttail call produces return_call.
define swifttailcc void @direct_tail_call(ptr swiftasync %ctx) {
; CHECK-LABEL: direct_tail_call:
; CHECK:         return_call direct_tail_call
  musttail call swifttailcc void @direct_tail_call(ptr swiftasync %ctx)
  ret void
}

; Indirect musttail call produces return_call_indirect.
define swifttailcc void @indirect_tail_call(ptr swiftasync %ctx, ptr %fn) {
; CHECK-LABEL: indirect_tail_call:
; CHECK:         return_call_indirect
  musttail call swifttailcc void %fn(ptr swiftasync %ctx)
  ret void
}

; Tail call with swiftasync and swiftself parameters.
; Note: swifterror is not allowed in swifttailcc musttail caller/callee.
; Errors are handled through the async context rather than through the
; swifterror register convention.
declare swifttailcc void @callee_swift_params(i32, ptr swiftasync, ptr swiftself)

define swifttailcc void @tail_call_swift_params(i32 %x, ptr swiftasync %ctx, ptr swiftself %self) {
; CHECK-LABEL: tail_call_swift_params:
; CHECK:         return_call callee_swift_params
  musttail call swifttailcc void @callee_swift_params(i32 %x, ptr swiftasync %ctx, ptr swiftself %self)
  ret void
}

; Non-tail call: regular call instruction, not return_call.
declare swifttailcc void @other_func(ptr swiftasync)
declare void @side_effect()

define swifttailcc void @regular_call(ptr swiftasync %ctx) {
; CHECK-LABEL: regular_call:
; CHECK:         call other_func
; CHECK:         call side_effect
; CHECK:         return
  call swifttailcc void @other_func(ptr swiftasync %ctx)
  call void @side_effect()
  ret void
}

; Tail call with return value.
declare swifttailcc i32 @callee_i32(ptr swiftasync, i32)

define swifttailcc i32 @with_return(ptr swiftasync %ctx, i32 %x) {
; CHECK-LABEL: with_return:
; CHECK:         return_call callee_i32
  %r = musttail call swifttailcc i32 @callee_i32(ptr swiftasync %ctx, i32 %x)
  ret i32 %r
}

; Mixed calling conventions: swifttailcc function calling a C function.
declare void @c_function(i32)

define swifttailcc void @mixed_cc(ptr swiftasync %ctx) {
; CHECK-LABEL: mixed_cc:
; CHECK:         call c_function
  call void @c_function(i32 42)
  ret void
}

; Signature padding: a swifttailcc function with no Swift attributes gets
; swiftself, swifterror, and swiftasync dummy params padded automatically.
define swifttailcc void @no_swift_params(i32 %x) {
; CHECK-LABEL: no_swift_params:
; CHECK:         .functype no_swift_params (i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  ret void
}

; Return type mismatch: advisory tail call falls back to regular call.
declare swifttailcc i32 @returns_i32_callee(ptr swiftasync)

define swifttailcc void @return_type_mismatch(ptr swiftasync %ctx) {
; CHECK-LABEL: return_type_mismatch:
; CHECK-NOT:     return_call
; CHECK:         call $drop=, returns_i32_callee
  tail call swifttailcc i32 @returns_i32_callee(ptr swiftasync %ctx)
  ret void
}

; Varargs callee: advisory tail call falls back to regular call.
declare swifttailcc void @varargs_callee(ptr, ...)

define swifttailcc void @varargs_tail(ptr swiftasync %ctx) {
; CHECK-LABEL: varargs_tail:
; CHECK-NOT:     return_call
; CHECK:         call varargs_callee
  tail call swifttailcc void @varargs_callee(ptr swiftasync %ctx)
  ret void
}

; Indirect call signature consistency: all indirect calls to a swifttailcc
; function must produce the same call_indirect signature, regardless of which
; swift parameter attributes are present at the call site. This also verifies
; that FixFunctionBitcasts skips swifttailcc (the IR type doesn't match the
; padded Wasm type, so without the skip a wrapper would break this).
define swifttailcc void @indirect_target(i32, i32) {
; CHECK-LABEL: indirect_target:
; CHECK:         .functype indirect_target (i32, i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  ret void
}
@fn_ptr = global ptr @indirect_target

define swifttailcc void @test_indirect_consistency() {
; CHECK-LABEL: test_indirect_consistency:
  %p = load ptr, ptr @fn_ptr

  ; No swift attrs — swiftself, swifterror, swiftasync all padded.
; CHECK: call_indirect __indirect_function_table, (i32, i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  call swifttailcc void %p(i32 1, i32 2)

  ; swiftasync present — swiftself and swifterror padded.
; CHECK: call_indirect __indirect_function_table, (i32, i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  call swifttailcc void %p(i32 1, i32 2, ptr swiftasync null)

  ; swiftself present — swifterror and swiftasync padded.
; CHECK: call_indirect __indirect_function_table, (i32, i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  call swifttailcc void %p(i32 1, i32 2, ptr swiftself null)

  ; swiftasync + swiftself present — swifterror padded.
; CHECK: call_indirect __indirect_function_table, (i32, i32, [[PTR]], [[PTR]], [[PTR]]) -> ()
  call swifttailcc void %p(i32 1, i32 2, ptr swiftasync null, ptr swiftself null)

  ret void
}
