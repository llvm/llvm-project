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

; CHECK: define i32 @load(ptr %a)
; CHECK:     %i = load i32, ptr %a
; CHECK:     ret i32 %i
define i32 @load(ptr %a) {
    %i = load i32, ptr %a
    ret i32 %i
}

; CHECK: define void @store(ptr %a, i32 %i)
; CHECK:     store i32 %i, ptr %a
; CHECK:     ret void
define void @store(ptr %a, i32 %i) {
    store i32 %i, ptr %a
    ret void
}

; CHECK: define void @gep(ptr %a)
; CHECK:     %b = getelementptr i8, ptr %a, i32 2
; CHECK:     ret void
define void @gep(ptr %a) {
    %b = getelementptr i8, ptr %a, i32 2
    ret void
}

; CHECK: define void @cmpxchg(ptr %p, i32 %a, i32 %b)
; CHECK:     %val_success = cmpxchg ptr %p, i32 %a, i32 %b acq_rel monotonic
; CHECK:     ret void
define void @cmpxchg(ptr %p, i32 %a, i32 %b) {
    %val_success = cmpxchg ptr %p, i32 %a, i32 %b acq_rel monotonic
    ret void
}

; CHECK: define void @atomicrmw(ptr %a, i32 %i)
; CHECK:     %b = atomicrmw add ptr %a, i32 %i acquire
; CHECK:     ret void
define void @atomicrmw(ptr %a, i32 %i) {
    %b = atomicrmw add ptr %a, i32 %i acquire
    ret void
}
