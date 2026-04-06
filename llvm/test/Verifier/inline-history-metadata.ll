; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

@x = global i32 0

define void @f() {
  ret void
}

; CHECK: !inline_history should only exist on calls
define void @wrong_instr(ptr %x) {
  load ptr, ptr %x, !inline_history !{ptr @wrong_instr}
  ret void
}

; CHECK: !inline_history operands must be functions or null
define void @global_value_operand() {
  call void @f(), !inline_history !{ptr @x}
  ret void
}

; CHECK: !inline_history operands must be functions or null
define void @metadata_operand() {
  call void @f(), !inline_history !{!0}
  ret void
}

; CHECK: !inline_history operands must be functions or null
define void @nullptr_operand() {
  call void @f(), !inline_history !{ptr null}
  ret void
}

; CHECK-NOT: !inline_history operands must be functions or null

define void @empty_metadata() {
  call void @f(), !inline_history !{}
  ret void
}

define void @null_metadata() {
  call void @f(), !inline_history !{null}
  ret void
}

define void @function_metadata() {
  call void @f(), !inline_history !{ptr @f}
  ret void
}

define void @mixed_metadata() {
  call void @f(), !inline_history !{null, ptr @f, null, ptr @mixed_metadata}
  ret void
}

!0 = !{}
