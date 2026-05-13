; RUN: not opt < %s -passes=instrumentor -instrumentor-read-config-file=%S/bad_function_regex.json -S 2>&1 | FileCheck %s

; CHECK: error: failed to parse function regex: repetition-operator operand invalid

define i32 @foo() {
entry:
  %0 = alloca i32, align 4
  store i32 0, ptr %0, align 4
  %2 = load i32, ptr %0, align 4
  ret i32 %2
}
