; RUN: opt -passes=verify -S < %s | FileCheck %s

; CHECK: @load
define i32 @load(ptr %a) {
    %i = load i32, ptr %a
    ret i32 %i
}

; CHECK: @store
define void @store(ptr %a, i32 %i) {
    store i32 %i, ptr %a
    ret void
}

; CHECK: @cmpxchg
define void @cmpxchg(ptr %p, i32 %a, i32 %b) {
    %val_success = cmpxchg ptr %p, i32 %a, i32 %b acq_rel monotonic
    ret void
}

; CHECK: @atomicrmw
define void @atomicrmw(ptr %a, i32 %i) {
    %b = atomicrmw add ptr %a, i32 %i acquire
    ret void
}

define void @opaque_mangle(ptr %a) {
    call void @llvm.lifetime.start.p0(i64 8, ptr %a)
    call void @llvm.lifetime.end.p0(i64 8, ptr %a)
    ret void
}

; CHECK: @llvm.lifetime.start.p0
; CHECK: @llvm.lifetime.end.p0
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
