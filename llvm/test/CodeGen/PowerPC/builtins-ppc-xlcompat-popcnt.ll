; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s --check-prefix=CHECK-64B
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-64B
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-32B
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-64B

@ui = external global i32, align 4
@ull = external global i64, align 8

define dso_local signext i32 @test_builtin_ppc_poppar4() {
; CHECK-32B-LABEL: test_builtin_ppc_poppar4:
; CHECK-32B:         popcntw 3, 3
; CHECK-32B-NEXT:    clrlwi 3, 3, 31
; CHECK-32B-NEXT:    blr
; CHECK-64B-LABEL: test_builtin_ppc_poppar4:
; CHECK-64B:         popcntw 3, 3
; CHECK-64B-NEXT:    clrlwi 3, 3, 31
; CHECK-64B-NEXT:    blr
entry:
  %0 = load i32, i32* @ui, align 4
  %1 = load i32, i32* @ui, align 4
  %2 = call i32 @llvm.ctpop.i32(i32 %1)
  %3 = and i32 %2, 1
  ret i32 %3
}

declare i32 @llvm.ctpop.i32(i32)

define dso_local signext i32 @test_builtin_ppc_poppar8() {
; CHECK-32B-LABEL: test_builtin_ppc_poppar8:
; CHECK-32B:         xor 3, 3, 4
; CHECK-32B-NEXT:    popcntw 3, 3
; CHECK-32B-NEXT:    clrlwi 3, 3, 31
; CHECK-32B-NEXT:    blr
; CHECK-64B-LABEL: test_builtin_ppc_poppar8:
; CHECK-64B:         popcntd 3, 3
; CHECK-64B-NEXT:    clrldi 3, 3, 63
; CHECK-64B-NEXT:    blr
entry:
  %0 = load i64, i64* @ull, align 8
  %1 = load i64, i64* @ull, align 8
  %2 = call i64 @llvm.ctpop.i64(i64 %1)
  %3 = and i64 %2, 1
  %cast = trunc i64 %3 to i32
  ret i32 %cast
}

declare i64 @llvm.ctpop.i64(i64)
