; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @foo() !prof !0 {
  unreachable
}

; CHECK: function must have a single !prof attachment
define void @foo2() !prof !0 !prof !0 {
  unreachable
}

!0 = !{}
