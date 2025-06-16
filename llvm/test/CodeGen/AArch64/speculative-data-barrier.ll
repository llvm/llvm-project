; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

; CHECK-LABEL:  f:
; CHECK:        %bb.0:
; CHECK-NEXT:       csdb
; CHECK-NEXT:       ret
define dso_local void @f() {
  call void @llvm.speculative.data.barrier()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.speculative.data.barrier() #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }
