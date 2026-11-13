; RUN: not opt -passes=verify %s -disable-output 2>&1 | FileCheck %s
define void @foo() safestack "unsafe-stack-size"="16" { ret void }
define void @bar() safestack "unsafe-stack-size"="-1" { ret void }
define void @baz() safestack "unsafe-stack-size"="not-an-integer" { ret void }

; CHECK-NOT: "unsafe-stack-size" takes an unsigned integer: 16
; CHECK: "unsafe-stack-size" takes an unsigned integer: -1
; CHECK: "unsafe-stack-size" takes an unsigned integer: not-an-integer
