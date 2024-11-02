; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: expected uint32 param
define void @f(target("type", i32, 0, void) %a) {
    ret void
}
