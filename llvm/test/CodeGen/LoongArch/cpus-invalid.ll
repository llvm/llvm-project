; RUN: llc < %s --mtriple=loongarch64 --mattr=+64bit --mcpu=invalidcpu 2>&1 | FileCheck %s

; CHECK: {{.*}} is not a recognized processor for this target

define void @f() {
  ret void
}
