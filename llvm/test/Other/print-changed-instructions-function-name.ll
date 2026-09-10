; RUN: opt -passes=name-anon-globals -disable-output -print-changed=inst-quiet %s 2>&1 | FileCheck %s --allow-empty

define void @0() {
entry:
  ret void
}

; CHECK-NOT: IR Instruction Changes
