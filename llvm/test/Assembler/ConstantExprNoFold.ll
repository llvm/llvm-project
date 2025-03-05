; This test checks to make sure that constant exprs don't fold in some simple
; situations

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Even give it a datalayout, to tempt folding as much as possible.
target datalayout = "p:32:32"

@A = global i64 0
@B = global i64 0

; CHECK: @E = global ptr addrspace(1) addrspacecast (ptr @A to ptr addrspace(1))
@E = global ptr addrspace(1) addrspacecast(ptr @A to ptr addrspace(1))

; Don't add an inbounds on @weak.gep, since @weak may be null.
; CHECK: @weak.gep = global ptr getelementptr (i32, ptr @weak, i32 1)
@weak.gep = global ptr getelementptr (i32, ptr @weak, i32 1)
@weak = extern_weak global i32

; Don't add an inbounds on @glob.a3, since it's not inbounds.
; CHECK: @glob.a3 = alias i32, getelementptr (i32, ptr @glob.a2, i32 1)
@glob = global i32 0
@glob.a3 = alias i32, getelementptr (i32, ptr @glob.a2, i32 1)
@glob.a2 = alias i32, getelementptr (i32, ptr @glob.a1, i32 1)
@glob.a1 = alias i32, ptr @glob
