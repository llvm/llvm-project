; RUN: llc < %s --mtriple=xtensa --mcpu=invalid 2>&1 | FileCheck %s

; CHECK: {{.*}} is not a recognized processor for this target

define void @f() {
  ret void
}

