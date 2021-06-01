; RUN: llc -verify-machineinstrs -mtriple=powerpcle-unknown-linux-gnu \
; RUN: -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu \
; RUN: -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN: -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN: -mcpu=pwr8 < %s | FileCheck %s

define dso_local void @test_builtin_ppc_eieio() #0 {
; CHECK-LABEL: test_builtin_ppc_eieio

entry:
  call void @llvm.ppc.eieio()
; CHECK: ori 2, 2, 0
; CHECK-NEXT: ori 2, 2, 0
; CHECK-NEXT: eieio
 
  ret void
}

declare void @llvm.ppc.eieio() #2

define dso_local void @test_builtin_ppc_iospace_eieio() #0 {
; CHECK-LABEL: test_builtin_ppc_iospace_eieio

entry:
  call void @llvm.ppc.iospace.eieio()
; CHECK: ori 2, 2, 0
; CHECK-NEXT: ori 2, 2, 0
; CHECK-NEXT: eieio
 
  ret void
}

declare void @llvm.ppc.iospace.eieio() #2

define dso_local void @test_builtin_ppc_iospace_lwsync() #0 {
; CHECK-LABEL: test_builtin_ppc_iospace_lwsync

entry:
  call void @llvm.ppc.iospace.lwsync()
; CHECK: lwsync

  ret void
}

declare void @llvm.ppc.iospace.lwsync() #2

define dso_local void @test_builtin_ppc_iospace_sync() #0 {
; CHECK-LABEL: test_builtin_ppc_iospace_sync

entry:
  call void @llvm.ppc.iospace.sync()
; CHECK: sync

  ret void
}

declare void @llvm.ppc.iospace.sync() #2

define dso_local void @test_builtin_ppc_icbt() #0 {
; CHECK-LABEL: test_builtin_ppc_icbt

entry:
  %a = alloca i8*, align 8
  %0 = load i8*, i8** %a, align 8
  call void @llvm.ppc.icbt(i8* %0)
; CHECK: icbt 0, 0, 3

  ret void
}

declare void @llvm.ppc.icbt(i8*) #2
