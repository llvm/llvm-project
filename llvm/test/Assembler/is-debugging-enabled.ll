; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define i1 @query_debugging_enabled() {
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret i1 %enabled
}

; CHECK: declare noundef i1 @llvm.is.debugging.enabled() #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = { nocallback nofree nomerge nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

declare i1 @llvm.is.debugging.enabled()
