; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: !mem.cache_hint is only valid on memory operations
define void @non_memory_op(i32 %x, i32 %y) {
  %z = add i32 %x, %y, !mem.cache_hint !{!{!"operand_no", i32 0}}
  ret void
}

; CHECK: !mem.cache_hint operand must not be null
define void @null_operand(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{null}
  ret void
}

; CHECK: !mem.cache_hint operand must be a metadata node
define void @operand_not_mdnode(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{!"not_a_node"}
  ret void
}

; CHECK: !mem.cache_hint node must have even number of operands
define void @odd_operands(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{!{!"operand_no", i32 0, !"extra"}}
  ret void
}

; CHECK: !mem.cache_hint key must be a string
define void @key_not_string(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{!{i32 0, i32 1}}
  ret void
}

; CHECK: !mem.cache_hint 'operand_no' must be an integer constant
define void @operand_no_not_integer(ptr %p) {
  %v = load i32, ptr %p, !mem.cache_hint !{!{!"operand_no", !"zero"}}
  ret void
}
