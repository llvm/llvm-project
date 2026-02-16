; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

declare void @foo(i32, i32)
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

; CHECK: !mem.cache_hint operand_no must be an integer constant in pair
define void @operand_no_not_integer(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{!"zero", !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint operand_no must refer to a valid memory object operand
define void @operand_no_not_pointer(i32 %x, i32 %y) {
  call void @foo(i32 %x, i32 %y), !mem.cache_hint !{i32 0, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint operand_no must refer to a valid memory object operand
define void @operand_no_out_of_range(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 1, !{!"nvvm.l1_eviction", !"first"}}
  ret void
}

; CHECK: !mem.cache_hint contains duplicate operand_no
define void @duplicate_operand_no(ptr %p) {
  call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %p, i64 8, i1 false), !mem.cache_hint !{
      i32 0, !{!"nvvm.l1_eviction", !"first"},
      i32 0, !{!"nvvm.l1_eviction", !"last"}}
  ret void
}

; CHECK: !mem.cache_hint hint node must be a metadata node
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

; CHECK: !mem.cache_hint hint node contains duplicate key
define void @duplicate_key(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{i32 0, !{
      !"nvvm.l1_eviction", !"first",
      !"nvvm.l1_eviction", !"last"}}
  ret void
}
