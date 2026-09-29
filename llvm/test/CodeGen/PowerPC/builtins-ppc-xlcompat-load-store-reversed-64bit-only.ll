; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s

@ull = external global i64, align 8
@ull_addr = external global ptr, align 8

define dso_local void @test_builtin_ppc_store8r() {
; CHECK-LABEL: test_builtin_ppc_store8r:
; CHECK:         stdbrx 3, 0, 4
; CHECK-NEXT:    blr
;
entry:
  %0 = load i64, ptr @ull, align 8
  %1 = load ptr, ptr @ull_addr, align 8
  call void @llvm.ppc.store8r(i64 %0, ptr %1)
  ret void
}

declare void @llvm.ppc.store8r(i64, ptr)

define dso_local i64 @test_builtin_ppc_load8r() {
; CHECK-LABEL: test_builtin_ppc_load8r:
; CHECK:         ldbrx 3, 0, 3
; CHECK-NEXT:    blr
entry:
  %0 = load ptr, ptr @ull_addr, align 8
  %1 = call i64 @llvm.ppc.load8r(ptr %0)
  ret i64 %1
}

declare i64 @llvm.ppc.load8r(ptr)
