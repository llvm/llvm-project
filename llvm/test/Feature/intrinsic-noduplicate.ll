; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; REQUIRES: nvptx-registered-target

; Make sure LLVM knows about the convergent attribute on the
; llvm.nvvm.barrier.cta.sync.aligned.all intrinsic.

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)

; CHECK: declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #[[ATTRNUM:[0-9]+]]
; CHECK: attributes #[[ATTRNUM]] = { convergent nocallback nounwind }
