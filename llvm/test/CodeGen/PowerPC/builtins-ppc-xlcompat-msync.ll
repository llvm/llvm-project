; RUN: llc -verify-machineinstrs -mtriple=powerpcle-unknown-linux-gnu \
; RUN:    -mattr=+msync -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu \
; RUN:    -mattr=+msync -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:    -mattr=+msync -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:    -mattr=+msync -mcpu=pwr8 < %s | FileCheck %s

define dso_local void @test_builtin_ppc_iospace_lwsync() #0 {
; CHECK-LABEL: test_builtin_ppc_iospace_lwsync

entry:
  call void @llvm.ppc.iospace.lwsync()
; CHECK: msync

  ret void
}

declare void @llvm.ppc.iospace.lwsync() #2

define dso_local void @test_builtin_ppc_iospace_sync() #0 {
; CHECK-LABEL: test_builtin_ppc_iospace_sync

entry:
  call void @llvm.ppc.iospace.sync()
; CHECK: msync

  ret void
}

declare void @llvm.ppc.iospace.sync() #2

