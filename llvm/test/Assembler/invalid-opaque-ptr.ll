; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: pointer to this type is invalid
define void @f(ptr %a) {
    %b = bitcast ptr %a to ptr*
    ret void
}
