; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode" -S | FileCheck %s

@a = global i32 1

; CHECK: @a = global i32 1
;; Make sure the module is in the correct section.
; CHECK: @llvm.embedded.object = private constant {{.*}}, section ".llvm.lto", align 1

;; Ensure that the metadata is in llvm.compiler.used.
; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @llvm.embedded.object], section "llvm.metadata"

;; Make sure the metadata correlates to the .llvm.lto section.
; CHECK: !llvm.embedded.objects = !{!1}
; CHECK: !0 = !{}
; CHECK: !{ptr @llvm.embedded.object, !".llvm.lto"}
