; RUN: split-file %s %t
; RUN: not llubi --verbose < %t/array_count_overflow.ll 2>&1 | FileCheck %t/array_count_overflow.ll
; RUN: not llubi --verbose < %t/array_count_truncation.ll 2>&1 | FileCheck %t/array_count_truncation.ll
; RUN: not llubi --verbose < %t/global_unsized.ll 2>&1 | FileCheck %t/global_unsized.ll
; RUN: not llubi --verbose < %t/load_target_ext.ll 2>&1 | FileCheck %t/load_target_ext.ll
; RUN: not llubi --verbose < %t/shufflevector_large_size.ll 2>&1 | FileCheck %t/shufflevector_large_size.ll
; RUN: not llubi --verbose < %t/target_ext_extractvalue.ll 2>&1 | FileCheck %t/target_ext_extractvalue.ll
; RUN: not llubi --verbose < %t/target_ext_freeze_poison.ll 2>&1 | FileCheck %t/target_ext_freeze_poison.ll
; RUN: not llubi --verbose < %t/target_ext_global.ll 2>&1 | FileCheck %t/target_ext_global.ll
; RUN: not llubi --verbose < %t/target_ext_indirect_call.ll 2>&1 | FileCheck %t/target_ext_indirect_call.ll
; RUN: not llubi --verbose < %t/target_ext_main_arg.ll 2>&1 | FileCheck %t/target_ext_main_arg.ll
; RUN: not llubi --verbose < %t/target_ext_return.ll 2>&1 | FileCheck %t/target_ext_return.ll
; RUN: not llubi --verbose < %t/target_ext_store.ll 2>&1 | FileCheck %t/target_ext_store.ll
; RUN: not llubi --verbose < %t/target_ext_store_array.ll 2>&1 | FileCheck %t/target_ext_store_array.ll
; RUN: not llubi --verbose < %t/target_ext_store_nested_aggregate.ll 2>&1 | FileCheck %t/target_ext_store_nested_aggregate.ll
; RUN: not llubi --verbose < %t/target_ext_store_struct.ll 2>&1 | FileCheck %t/target_ext_store_struct.ll
; RUN: not llubi --verbose < %t/target_ext_store_vector_element.ll 2>&1 | FileCheck %t/target_ext_store_vector_element.ll
; RUN: not llubi --verbose < %t/target_ext_vararg.ll 2>&1 | FileCheck %t/target_ext_vararg.ll
; RUN: not llubi --verbose < %t/token_constant.ll 2>&1 | FileCheck %t/token_constant.ll
; RUN: llubi --verbose < %t/unsized_types.ll 2>&1 | FileCheck %t/unsized_types.ll
; RUN: not llubi --verbose < %t/vector_reduce_fmax_empty.ll 2>&1 | FileCheck %t/vector_reduce_fmax_empty.ll
; RUN: not llubi --verbose < %t/vector_reduce_fmaximum_empty.ll 2>&1 | FileCheck %t/vector_reduce_fmaximum_empty.ll
; RUN: not llubi --verbose < %t/x86_amx_bitcast.ll 2>&1 | FileCheck %t/x86_amx_bitcast.ll
; RUN: not llubi --verbose < %t/x86_amx_store_struct.ll 2>&1 | FileCheck %t/x86_amx_store_struct.ll

;--- array_count_overflow.ll
define void @main() {
  %a = alloca [4 x [2147483648 x [2147483648 x i8]]], align 8
  %v = load [4 x [2147483648 x [2147483648 x i8]]], ptr %a
  ret void
}
; CHECK: [2147483648 x i8] is too large!
; CHECK-NEXT:   %v = load [4 x [2147483648 x [2147483648 x i8]]], ptr %a, align 1 at @main <stdin>:3
; CHECK-NEXT: error: -: input module cannot be executed!

;--- array_count_truncation.ll
@g = global i8 0
define void @main() {
  %v = load [4294967296 x [0 x i8]], ptr @g
  %e = extractvalue [4294967296 x [0 x i8]] %v, 0
  ret void
}
; CHECK: [4294967296 x [0 x i8]] is too large!
; CHECK-NEXT:   %v = load [4294967296 x [0 x i8]], ptr @g, align 1 at @main <stdin>:3
; CHECK-NEXT: error: -: input module cannot be executed!

;--- global_unsized.ll
%opaque = type opaque
@unsized = external global %opaque

define void @main() {
  ret void
}
; CHECK: Unsupported opaque struct type %opaque
; CHECK-NEXT:   at @unsized
; CHECK-NEXT: error: -: input module cannot be executed!

;--- load_target_ext.ll
define void @main() {
  %a = alloca i64
  %v = load target("spirv.Event"), ptr %a
  ret void
}
; CHECK: Unsupported type target("spirv.Event")
; CHECK-NEXT:   %v = load target("spirv.Event"), ptr %a, align 8 at @main <stdin>:3
; CHECK-NEXT: error: -: input module cannot be executed!

;--- shufflevector_large_size.ll
define void @main() {
  %x = shufflevector <vscale x 1073741823 x i32> poison, <vscale x 1073741823 x i32> poison, <vscale x 1 x i32> zeroinitializer
  ret void
}
; CHECK: <vscale x 1073741823 x i32> is too large!
; CHECK-NEXT:   %x = shufflevector <vscale x 1073741823 x i32> poison, <vscale x 1073741823 x i32> poison, <vscale x 1 x i32> zeroinitializer at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_extractvalue.ll
define void @main() {
  %x = extractvalue {i32, target("wasm.externref")} undef, 0
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   %x = extractvalue { i32, target("wasm.externref") } undef, 0 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_freeze_poison.ll
define void @main() {
  %x = freeze target("wasm.externref") poison
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   %x = freeze target("wasm.externref") poison at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_global.ll
@g = global target("wasm.externref") poison

define i32 @main() {
  ret i32 0
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   at @g
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_indirect_call.ll
define void @main() {
  call void null(target("wasm.externref") poison)
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   call void null(target("wasm.externref") poison) at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_main_arg.ll
define i32 @main(target("wasm.externref") %a) {
  ret i32 0
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   target("wasm.externref") %a at @main <stdin>:1
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_return.ll
define target("wasm.externref") @main() {
  ret target("wasm.externref") poison
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   at @main <stdin>:1
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_store.ll
define void @main() {
  %a = alloca i8
  store target("wasm.externref") poison, ptr %a
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   store target("wasm.externref") poison, ptr %a, align 8 at @main <stdin>:3
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_store_array.ll
define void @main() {
  store [2 x target("wasm.externref")] poison, ptr null
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   store [2 x target("wasm.externref")] poison, ptr null, align 8 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_store_nested_aggregate.ll
define void @main() {
  store {i32, [2 x target("wasm.externref")]} poison, ptr null
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   store { i32, [2 x target("wasm.externref")] } poison, ptr null, align 8 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_store_struct.ll
define void @main() {
  store {i32, target("wasm.externref")} poison, ptr null
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   store { i32, target("wasm.externref") } poison, ptr null, align 8 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_store_vector_element.ll
define void @main() {
  store <2 x target("llvm.test.vectorelement")> poison, ptr null
  ret void
}
; CHECK: Unsupported type target("llvm.test.vectorelement")
; CHECK-NEXT:   store <2 x target("llvm.test.vectorelement")> poison, ptr null, align 8 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- target_ext_vararg.ll
@.str = private constant [3 x i8] c"%d\00"

declare i32 @printf(ptr, ...)

define void @main() {
  %r = call i32 (ptr, ...) @printf(ptr @.str, target("wasm.externref") poison)
  ret void
}
; CHECK: Unsupported type target("wasm.externref")
; CHECK-NEXT:   %r = call i32 (ptr, ...) @printf(ptr @.str, target("wasm.externref") poison) at @main <stdin>:6
; CHECK-NEXT: error: -: input module cannot be executed!

;--- token_constant.ll
define void @main() {
  call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}
; CHECK: Unsupported type token
; CHECK-NEXT:   %1 = call i8 @llvm.coro.suspend(token none, i1 false) at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- unsized_types.ll
; Make sure we don't crash when handling unsized struct types in isValid.

define { i1, <vscale x 1 x i1> } @f() {
  ret { i1, <vscale x 1 x i1> } zeroinitializer
}

define void @main() {
  call { i1, <vscale x 1 x i1> } @f()
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Entering function: f
; CHECK-NEXT:   ret { i1, <vscale x 1 x i1> } zeroinitializer
; CHECK-NEXT: Exiting function: f
; CHECK-NEXT:   %1 = call { i1, <vscale x 1 x i1> } @f() => { F, { F, F, F, F } }
; CHECK-NEXT:   ret void
; CHECK-NEXT: Exiting function: main

;--- vector_reduce_fmax_empty.ll
define void @main() {
  %r = call float @llvm.vector.reduce.fmax.nxv1073741824f32(<vscale x 1073741824 x float> zeroinitializer)
  ret void
}
; CHECK: <vscale x 1073741824 x float> is too large!
; CHECK-NEXT:   %r = call float @llvm.vector.reduce.fmax.nxv1073741824f32(<vscale x 1073741824 x float> zeroinitializer) at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- vector_reduce_fmaximum_empty.ll
define void @main() {
  %r = call float @llvm.vector.reduce.fmaximum.nxv1073741824f32(<vscale x 1073741824 x float> zeroinitializer)
  ret void
}
; CHECK: <vscale x 1073741824 x float> is too large!
; CHECK-NEXT:   %r = call float @llvm.vector.reduce.fmaximum.nxv1073741824f32(<vscale x 1073741824 x float> zeroinitializer) at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- x86_amx_bitcast.ll
define void @main() {
  %x = bitcast x86_amx poison to i8192
  ret void
}
; CHECK: Unsupported type x86_amx
; CHECK-NEXT:   %x = bitcast x86_amx poison to i8192 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- x86_amx_store_struct.ll
define void @main() {
  store {i32, x86_amx} poison, ptr null
  ret void
}
; CHECK: Unsupported type x86_amx
; CHECK-NEXT:   store { i32, x86_amx } poison, ptr null, align 64 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!
