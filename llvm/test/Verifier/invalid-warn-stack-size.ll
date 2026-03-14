; RUN: not opt -passes=verify %s -disable-output 2>&1 | FileCheck %s
define void @foo() "warn-stack-size"="42" { ret void }
define void @bar() "warn-stack-size"="-1" { ret void }
define void @baz() "warn-stack-size"="999999999999999999999" { ret void }
define void @qux() "warn-stack-size"="a lot lol" { ret void }

; CHECK-NOT: "warn-stack-size" takes an unsigned integer: 42
; CHECK: "warn-stack-size" takes an unsigned integer: -1
; CHECK: "warn-stack-size" takes an unsigned integer: 999999999999999999999
; CHECK: "warn-stack-size" takes an unsigned integer: a lot lol
