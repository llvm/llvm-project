; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

declare void @foo(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly, ptr noalias readonly, i64, i1 immarg)

; CHECK: !mem.cache_hint is only valid on memory operations
define void @non_memory_op(i32 %x, i32 %y) {
  %z = add i32 %x, %y, !mem.cache_hint !{i32 0, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint must have even number of operands
define void @odd_top_level_operands(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0}
  ret void
}

; CHECK: !mem.cache_hint must alternate between i32 operand numbers and metadata hint nodes
define void @operand_no_not_integer(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{!"zero", !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint operand number must be non-negative
define void @operand_no_negative(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 -1, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint is not supported on non-intrinsic calls
define void @non_intrinsic_call(ptr %p) {
  call void @foo(ptr %p), !mem.cache_hint !{i32 0, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint operand number must refer to a pointer operand
define void @operand_no_not_pointer(ptr %d, ptr %s) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 8, i1 false), !mem.cache_hint !{i32 2, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint operand number must refer to a pointer operand
define void @store_operand_no_is_value(ptr %p) {
  store i32 0, ptr %p, !mem.cache_hint !{i32 0, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint operand number is out of range
define void @operand_no_out_of_range(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 1, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint contains duplicate operand number
define void @duplicate_operand_no(ptr %p) {
  call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %p, i64 8, i1 false), !mem.cache_hint !{
      i32 0, !{!"nvvm.l1_eviction", !"first"},
      i32 0, !{!"nvvm.l1_eviction", !"last"}}
  ret void
}

; CHECK: !mem.cache_hint operand numbers must be in increasing order
define void @operand_no_not_increasing(ptr %d, ptr %s) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 8, i1 false), !mem.cache_hint !{
      i32 1, !{!"nvvm.l1_eviction", !"first"},
      i32 0, !{!"nvvm.l1_eviction", !"last"}}
  ret void
}

; CHECK: !mem.cache_hint must alternate between i32 operand numbers and metadata hint nodes
define void @hint_node_not_mdnode(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0, !"not_a_node"}
  ret void
}

; CHECK: !mem.cache_hint hint node must have even number of operands
define void @hint_node_odd_operands(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0, !{!"nvvm.l1_eviction"}}
  ret void
}

; CHECK: !mem.cache_hint key must be a string
define void @key_not_string(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0, !{i32 0, !"first"}}
  ret void
}

; CHECK: !mem.cache_hint value must be a string or integer
define void @value_not_string_or_integer(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0, !{!"nvvm.l1_eviction", !{}}}
  ret void
}

; CHECK: !mem.cache_hint hint node contains duplicate key
define void @duplicate_key(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0, !{
      !"nvvm.l1_eviction", !"first",
      !"nvvm.l1_eviction", !"last"}}
  ret void
}
