; RUN: not llvm-as < %s -disable-output --opaque-pointers=0 2>&1 | FileCheck %s

; CHECK: warning: ptr type is only supported in -opaque-pointers mode
; CHECK: error: expected type
define void @f(ptr %a) {
    ret void
}
