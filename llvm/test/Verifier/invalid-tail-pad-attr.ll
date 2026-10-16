; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: "tail-pad-to-size" takes an unsigned integer:
; CHECK: "tail-pad-value" takes an unsigned integer:
define void @f() "tail-pad-to-size" "tail-pad-value" { ret void }
; CHECK: "tail-pad-to-size" takes an unsigned integer: a
; CHECK: "tail-pad-value" takes an unsigned integer: a
define void @fa() "tail-pad-to-size"="a" "tail-pad-value"="a" { ret void }
; CHECK: "tail-pad-to-size" takes an unsigned integer: -1
; CHECK: "tail-pad-value" takes an unsigned integer: -1
define void @f_1() "tail-pad-to-size"="-1" "tail-pad-value"="-1" { ret void }
; CHECK: "tail-pad-to-size" takes an unsigned integer: 3,
; CHECK: "tail-pad-value" takes an unsigned integer: 3,
define void @f3comma() "tail-pad-to-size"="3," "tail-pad-value"="3," { ret void }

; CHECK-NOT: takes an unsigned integer
define void @s1() "tail-pad-to-size"="1" "tail-pad-value"="1" { ret void }
