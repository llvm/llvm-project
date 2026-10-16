; RUN: split-file %s %t
; RUN: not llubi --verbose < %t/alloca_array_count_overflow.ll 2>&1 | FileCheck %t/alloca_array_count_overflow.ll
; RUN: not llubi --verbose < %t/alloca_type_size_wrap_wrong_result.ll 2>&1 | FileCheck %t/alloca_type_size_wrap_wrong_result.ll
; RUN: not llubi --verbose < %t/array_count_truncation.ll 2>&1 | FileCheck %t/array_count_truncation.ll
; RUN: not llubi --verbose < %t/cast_huge_operand_unvalidated_oom.ll 2>&1 | FileCheck %t/cast_huge_operand_unvalidated_oom.ll
; RUN: not llubi --verbose < %t/ce_shuffle_global.ll 2>&1 | FileCheck %t/ce_shuffle_global.ll
; RUN: not llubi --verbose < %t/fcmp_huge_operand.ll 2>&1 | FileCheck %t/fcmp_huge_operand.ll
; RUN: not llubi --verbose < %t/fptrunc_huge_operand.ll 2>&1 | FileCheck %t/fptrunc_huge_operand.ll
; RUN: not llubi --verbose < %t/global_unsized.ll 2>&1 | FileCheck %t/global_unsized.ll
; RUN: not llubi --verbose < %t/huge_scalable_ce_shuffle_crash.ll 2>&1 | FileCheck %t/huge_scalable_ce_shuffle_crash.ll
; RUN: not llubi --verbose < %t/icmp_huge_operand_unvalidated_oom.ll 2>&1 | FileCheck %t/icmp_huge_operand_unvalidated_oom.ll
; RUN: not llubi --verbose < %t/load_array_count_overflow.ll 2>&1 | FileCheck %t/load_array_count_overflow.ll
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
; RUN: not llubi --verbose < %t/vector_align_validation_bypass_oom.ll 2>&1 | FileCheck %t/vector_align_validation_bypass_oom.ll
; RUN: not llubi --verbose < %t/vector_reduce_fmax_empty.ll 2>&1 | FileCheck %t/vector_reduce_fmax_empty.ll
; RUN: not llubi --verbose < %t/vector_reduce_fmaximum_empty.ll 2>&1 | FileCheck %t/vector_reduce_fmaximum_empty.ll
; RUN: not llubi --verbose < %t/x86_amx_bitcast.ll 2>&1 | FileCheck %t/x86_amx_bitcast.ll
; RUN: not llubi --verbose < %t/x86_amx_store_struct.ll 2>&1 | FileCheck %t/x86_amx_store_struct.ll

;--- alloca_array_count_overflow.ll
define void @main() {
  %a = alloca [4 x [2147483648 x [2147483648 x i8]]], align 8
  ret void
}
; CHECK: [2147483648 x i8] is too large!
; CHECK-NEXT:   %a = alloca [4 x [2147483648 x [2147483648 x i8]]], align 8 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- alloca_type_size_wrap_wrong_result.ll
define void @main() {
  %a = alloca [4611686018427387904 x i32]
  ret void
}
; CHECK: [4611686018427387904 x i32] is too large!
; CHECK-NEXT:   %a = alloca [4611686018427387904 x i32], align 4 at @main <stdin>:2
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

;--- cast_huge_operand_unvalidated_oom.ll
define void @main() {
  %c = trunc <268435456 x i64> poison to <268435456 x i1>
  ret void
}
; CHECK: <268435456 x i64> is too large!
; CHECK-NEXT:   %c = trunc <268435456 x i64> poison to <268435456 x i1> at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- ce_shuffle_global.ll
@g = global float extractelement (<vscale x 4 x float> shufflevector (<vscale x 536870912 x float> splat (float 7.0), <vscale x 536870912 x float> poison, <vscale x 4 x i32> zeroinitializer), i32 0)
define void @main() { ret void }
; CHECK: <vscale x 536870912 x float> is too large!
; CHECK-NEXT:   at @g
; CHECK-NEXT: error: -: input module cannot be executed!

;--- fcmp_huge_operand.ll
define void @main() {
  %c = fcmp oeq <268435456 x double> poison, poison
  ret void
}
; CHECK: <268435456 x double> is too large!
; CHECK-NEXT:   %c = fcmp oeq <268435456 x double> poison, poison at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- fptrunc_huge_operand.ll
define void @main() {
  %c = fptrunc <67108864 x double> poison to <67108864 x float>
  ret void
}
; CHECK: <67108864 x double> is too large!
; CHECK-NEXT:   %c = fptrunc <67108864 x double> poison to <67108864 x float> at @main <stdin>:2
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

;--- huge_scalable_ce_shuffle_crash.ll
define void @main() {
  %v = extractelement <vscale x 4 x float> shufflevector (<vscale x 536870912 x float> splat (float 7.0), <vscale x 536870912 x float> poison, <vscale x 4 x i32> zeroinitializer), i32 0
  ret void
}
; CHECK: <vscale x 536870912 x float> is too large!
; CHECK-NEXT:   %v = extractelement <vscale x 4 x float> shufflevector (<vscale x 536870912 x float> splat (float 7.000000e+00), <vscale x 536870912 x float> poison, <vscale x 4 x i32> zeroinitializer), i32 0 at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- icmp_huge_operand_unvalidated_oom.ll
define void @main() {
  %c = icmp eq <268435456 x i64> poison, poison
  ret void
}
; CHECK: <268435456 x i64> is too large!
; CHECK-NEXT:   %c = icmp eq <268435456 x i64> poison, poison at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- load_array_count_overflow.ll
define void @main() {
  %v = load [4 x [2147483648 x [2147483648 x i8]]], ptr null
  ret void
}
; CHECK: [2147483648 x i8] is too large!
; CHECK-NEXT:   %v = load [4 x [2147483648 x [2147483648 x i8]]], ptr null, align 1 at @main <stdin>:2
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
define void @main() {
  ret void
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
define void @main(target("wasm.externref") %a) {
  ret void
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
; CHECK-NEXT:   %r = call i32 (ptr, ...) @printf(ptr @.str, target("wasm.externref") poison) at @main <stdin>:4
; CHECK-NEXT: error: -: input module cannot be executed!

;--- token_constant.ll
define void @main() {
  call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}
; CHECK: Unsupported type token
; CHECK-NEXT:   %1 = call i8 @llvm.coro.suspend(token none, i1 false) at @main <stdin>:2
; CHECK-NEXT: error: -: input module cannot be executed!

;--- vector_align_validation_bypass_oom.ll
target datalayout = "v1:32768"
@g = global [1048575 x <1 x i1>] zeroinitializer
define void @main() { ret void }
; CHECK: [1048575 x <1 x i1>] is too large!
; CHECK-NEXT:   at @g
; CHECK-NEXT: error: -: input module cannot be executed!

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
