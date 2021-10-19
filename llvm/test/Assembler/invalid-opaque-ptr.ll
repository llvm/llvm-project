; RUN: not llvm-as < %s -opaque-pointers -disable-output 2>&1 | FileCheck %s

; CHECK: ptr* is invalid - use ptr instead
define void @f(ptr %a) {
    %b = bitcast ptr %a to ptr*
    ret void
}
