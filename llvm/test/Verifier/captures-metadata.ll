; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: !captures metadata can only be applied to store instructions
define void @wrong_instr_type(ptr %x) {
  load ptr, ptr %x, !captures !{!"address"}
  ret void
}

; CHECK: captures metadata can only be applied to store with value operand of pointer type
define void @wrong_op_type(i32 %x, ptr %y) {
  store i32 %x, ptr %y, !captures !{!"address"}
  ret void
}

; CHECK: !captures metadata cannot be empty
define void @empty(ptr %x, ptr %y) {
  store ptr %x, ptr %y, !captures !{}
  ret void
}

; CHECK: !captures metadata must be a list of strings
define void @not_string(ptr %x, ptr %y) {
  store ptr %x, ptr %y, !captures !{!{}}
  ret void
}

; CHECK: invalid entry in !captures metadata
define void @invalid_str(ptr %x, ptr %y) {
  store ptr %x, ptr %y, !captures !{!"foo"}
  ret void
}

; CHECK: invalid entry in !captures metadata
define void @invalid_none(ptr %x, ptr %y) {
  store ptr %x, ptr %y, !captures !{!"none"}
  ret void
}
