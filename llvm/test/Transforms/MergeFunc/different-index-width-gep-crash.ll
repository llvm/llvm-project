; RUN: opt -passes=mergefunc -disable-output < %s

target datalayout = "e-p:64:64-p2:128:128:128:32"

define void @foo(ptr addrspace(2) %x) {
    %tmp = getelementptr i32, ptr addrspace(2) %x, i32 1
    ret void
}

define void @bar(ptr addrspace(2) %x) {
    %tmp = getelementptr i32, ptr addrspace(2) %x, i32 1
    ret void
}
