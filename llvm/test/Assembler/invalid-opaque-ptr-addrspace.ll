; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: ptr* is invalid - use ptr instead
define void @f(ptr addrspace(3) %a) {
    %b = bitcast ptr addrspace(3) %a to ptr addrspace(3)*
    ret void
}
