; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; RUN: llc -mtriple=powerpc64-unknown-freebsd -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc-unknown-freebsd -verify-machineinstrs < %s | FileCheck %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs < %s | FileCheck %s

; CHECK-NOT: vrsave
; CHECK-NOT: mfspr
; CHECK-NOT: mtspr

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  call void @llvm.eh.unwind.init()
  ret void
}

; Function Attrs: nounwind
declare void @llvm.eh.unwind.init() #0

attributes #0 = { nounwind }
