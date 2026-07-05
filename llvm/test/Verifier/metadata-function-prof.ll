; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: function declaration may not have a !prof attachment
declare !prof !0 void @f1()

define void @f2() !prof !0 {
  unreachable
}

; CHECK: function must have a single !prof attachment
define void @f3() !prof !0 !prof !0 {
  unreachable
}

!0 = !{!"function_entry_count", i64 100}
