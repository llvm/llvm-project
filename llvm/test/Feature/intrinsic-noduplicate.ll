; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; REQUIRES: nvptx-registered-target

; Make sure LLVM knows about the convergent attribute on the
; llvm.nvvm.barrier0 intrinsic.

declare void @llvm.nvvm.barrier0()

; CHECK: declare void @llvm.nvvm.barrier0() #[[ATTRNUM:[0-9]+]]
; CHECK: attributes #[[ATTRNUM]] = { convergent nocallback nounwind }
