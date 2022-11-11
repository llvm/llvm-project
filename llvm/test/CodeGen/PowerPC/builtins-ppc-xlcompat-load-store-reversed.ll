; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s --check-prefix=CHECK-64B
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-64B
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-32B
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-64B

@us = external global i16, align 2
@us_addr = external global ptr, align 8
@ui = external global i32, align 4
@ui_addr = external global ptr, align 8

define dso_local void @test_builtin_ppc_store2r() {
; CHECK-64B-LABEL: test_builtin_ppc_store2r:
; CHECK-64B:         sthbrx 3, 0, 4
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_store2r:
; CHECK-32B:         sthbrx 3, 0, 4
; CHECK-32B-NEXT:    blr
entry:
  %0 = load i16, ptr @us, align 2
  %conv = zext i16 %0 to i32
  %1 = load ptr, ptr @us_addr, align 8
  call void @llvm.ppc.store2r(i32 %conv, ptr %1)
  ret void
}

declare void @llvm.ppc.store2r(i32, ptr)

define dso_local void @test_builtin_ppc_store4r() {
; CHECK-64B-LABEL: test_builtin_ppc_store4r:
; CHECK-64B:         stwbrx 3, 0, 4
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_store4r:
; CHECK-32B:         stwbrx 3, 0, 4
; CHECK-32B-NEXT:    blr
entry:
  %0 = load i32, ptr @ui, align 4
  %1 = load ptr, ptr @ui_addr, align 8
  call void @llvm.ppc.store4r(i32 %0, ptr %1)
  ret void
}

declare void @llvm.ppc.store4r(i32, ptr)

define dso_local zeroext i16 @test_builtin_ppc_load2r() {
; CHECK-64B-LABEL: test_builtin_ppc_load2r:
; CHECK-64B:         lhbrx 3, 0, 3
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_load2r:
; CHECK-32B:         lhbrx 3, 0, 3
; CHECK-32B-NEXT:    blr
entry:
  %0 = load ptr, ptr @us_addr, align 8
  %1 = call i32 @llvm.ppc.load2r(ptr %0)
  %conv = trunc i32 %1 to i16
  ret i16 %conv
}

define dso_local signext i16 @test_builtin_ppc_load2r_signext() {
; CHECK-64B-LABEL: test_builtin_ppc_load2r_signext:
; CHECK-64B:         lhbrx 3, 0, 3
; CHECK-64B-NEXT:    extsh 3, 3
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_load2r_signext:
; CHECK-32B:         lhbrx 3, 0, 3
; CHECK-32B-NEXT:    extsh 3, 3
; CHECK-32B-NEXT:    blr
entry:
  %0 = load ptr, ptr @us_addr, align 8
  %1 = call i32 @llvm.ppc.load2r(ptr %0)
  %conv = trunc i32 %1 to i16
  ret i16 %conv
}

declare i32 @llvm.ppc.load2r(ptr)

define dso_local zeroext i32 @test_builtin_ppc_load4r() {
; CHECK-64B-LABEL: test_builtin_ppc_load4r:
; CHECK-64B:         lwbrx 3, 0, 3
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_load4r:
; CHECK-32B:         lwbrx 3, 0, 3
; CHECK-32B-NEXT:    blr
entry:
  %0 = load ptr, ptr @ui_addr, align 8
  %1 = call i32 @llvm.ppc.load4r(ptr %0)
  ret i32 %1
}

declare i32 @llvm.ppc.load4r(ptr)
