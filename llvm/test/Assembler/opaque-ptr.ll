; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define ptr @f(ptr %a) {
; CHECK:     %b = bitcast ptr %a to ptr
; CHECK:     ret ptr %b
define ptr @f(ptr %a) {
    %b = bitcast ptr %a to ptr
    ret ptr %b
}

; CHECK: define ptr @g(ptr addrspace(2) %a) {
; CHECK:     %b = addrspacecast ptr addrspace(2) %a to ptr
; CHECK:     ret ptr %b
define ptr @g(ptr addrspace(2) %a) {
    %b = addrspacecast ptr addrspace(2) %a to ptr addrspace(0)
    ret ptr addrspace(0) %b
}

; CHECK: define ptr addrspace(2) @g2(ptr %a) {
; CHECK:     %b = addrspacecast ptr %a to ptr addrspace(2)
; CHECK:     ret ptr addrspace(2) %b
define ptr addrspace(2) @g2(ptr addrspace(0) %a) {
    %b = addrspacecast ptr addrspace(0) %a to ptr addrspace(2)
    ret ptr addrspace(2) %b
}
