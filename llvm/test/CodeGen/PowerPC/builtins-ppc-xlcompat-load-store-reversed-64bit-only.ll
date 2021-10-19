; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s

@ull = external global i64, align 8
@ull_addr = external global i64*, align 8

define dso_local void @test_builtin_ppc_store8r() {
; CHECK-LABEL: test_builtin_ppc_store8r:
; CHECK:         stdbrx 3, 0, 4
; CHECK-NEXT:    blr
;
entry:
  %0 = load i64, i64* @ull, align 8
  %1 = load i64*, i64** @ull_addr, align 8
  %2 = bitcast i64* %1 to i8*
  call void @llvm.ppc.store8r(i64 %0, i8* %2)
  ret void
}

declare void @llvm.ppc.store8r(i64, i8*)

define dso_local i64 @test_builtin_ppc_load8r() {
; CHECK-LABEL: test_builtin_ppc_load8r:
; CHECK:         ldbrx 3, 0, 3
; CHECK-NEXT:    blr
entry:
  %0 = load i64*, i64** @ull_addr, align 8
  %1 = bitcast i64* %0 to i8*
  %2 = call i64 @llvm.ppc.load8r(i8* %1)
  ret i64 %2
}

declare i64 @llvm.ppc.load8r(i8*)
