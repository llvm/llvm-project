; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: "patchable-function-entry" takes an unsigned integer:
; CHECK: "patchable-function-entry" takes an unsigned integer: a
; CHECK: "patchable-function-entry" takes an unsigned integer: -1
; CHECK: "patchable-function-entry" takes an unsigned integer: 3,

define void @f() "patchable-function-entry" { ret void }
define void @fa() "patchable-function-entry"="a" { ret void }
define void @f_1() "patchable-function-entry"="-1" { ret void }
define void @f3comma() "patchable-function-entry"="3," { ret void }

; CHECK: "patchable-function-prefix" takes an unsigned integer:
; CHECK: "patchable-function-prefix" takes an unsigned integer: a
; CHECK: "patchable-function-prefix" takes an unsigned integer: -1
; CHECK: "patchable-function-prefix" takes an unsigned integer: 3,

define void @g() "patchable-function-prefix" { ret void }
define void @ga() "patchable-function-prefix"="a" { ret void }
define void @g_1() "patchable-function-prefix"="-1" { ret void }
define void @g3comma() "patchable-function-prefix"="3," { ret void }
