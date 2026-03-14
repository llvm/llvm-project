; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: ptr* is invalid - use ptr instead
define void @f(ptr %a) {
    %b = bitcast ptr %a to ptr*
    ret void
}
