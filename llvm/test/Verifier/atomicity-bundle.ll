; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @f()

; The bundle is not restricted to any particular callee.
define void @ordinary_call() {
  call void @f() [ "atomicity"(metadata !"release", metadata !"agent") ]
  ret void
}

; CHECK: Multiple "atomicity" operand bundles
define void @multiple() {
  call void @f() [ "atomicity"(metadata !"release", metadata !"agent"), "atomicity"(metadata !"release", metadata !"agent") ]
  ret void
}

; CHECK: Expected exactly two "atomicity" bundle operands
define void @wrong_operand_count() {
  call void @f() [ "atomicity"(metadata !"release") ]
  ret void
}

; CHECK: "atomicity" ordering operand must be a metadata string naming an atomic ordering
define void @not_an_ordering() {
  call void @f() [ "atomicity"(metadata !"consume", metadata !"agent") ]
  ret void
}

; CHECK: "atomicity" ordering operand must be a metadata string naming an atomic ordering
define void @ordering_not_a_string() {
  call void @f() [ "atomicity"(i32 0, metadata !"agent") ]
  ret void
}

; CHECK: "atomicity" syncscope operand must be a metadata string
define void @scope_not_a_string() {
  call void @f() [ "atomicity"(metadata !"release", i32 0) ]
  ret void
}
