; RUN: llc -verify-machineinstrs -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s

; CHECK-LABEL:  f:
; CHECK:        %bb.0:
; CHECK-NEXT:       lfence
; CHECK-NEXT:       ret
define dso_local void @f() {
  call void @llvm.speculative.data.barrier()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.speculative.data.barrier() #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }
