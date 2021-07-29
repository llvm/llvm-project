; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s --check-prefix=CHECK-64B
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-64B
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-32B
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   -mcpu=pwr7 < %s | FileCheck %s --check-prefix=CHECK-64B

@us = external global i16, align 2
@us_addr = external global i16*, align 8
@ui = external global i32, align 4
@ui_addr = external global i32*, align 8

define dso_local void @test_builtin_ppc_store2r() {
; CHECK-64B-LABEL: test_builtin_ppc_store2r:
; CHECK-64B:         sthbrx 3, 0, 4
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_store2r:
; CHECK-32B:         sthbrx 3, 0, 4
; CHECK-32B-NEXT:    blr
entry:
  %0 = load i16, i16* @us, align 2
  %conv = zext i16 %0 to i32
  %1 = load i16*, i16** @us_addr, align 8
  %2 = bitcast i16* %1 to i8*
  call void @llvm.ppc.store2r(i32 %conv, i8* %2)
  ret void
}

declare void @llvm.ppc.store2r(i32, i8*)

define dso_local void @test_builtin_ppc_store4r() {
; CHECK-64B-LABEL: test_builtin_ppc_store4r:
; CHECK-64B:         stwbrx 3, 0, 4
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_store4r:
; CHECK-32B:         stwbrx 3, 0, 4
; CHECK-32B-NEXT:    blr
entry:
  %0 = load i32, i32* @ui, align 4
  %1 = load i32*, i32** @ui_addr, align 8
  %2 = bitcast i32* %1 to i8*
  call void @llvm.ppc.store4r(i32 %0, i8* %2)
  ret void
}

declare void @llvm.ppc.store4r(i32, i8*)

define dso_local zeroext i16 @test_builtin_ppc_load2r() {
; CHECK-64B-LABEL: test_builtin_ppc_load2r:
; CHECK-64B:         lhbrx 3, 0, 3
; CHECK-64B-NEXT:    clrldi 3, 3, 48
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_load2r:
; CHECK-32B:         lhbrx 3, 0, 3
; CHECK-32B-NEXT:    clrlwi 3, 3, 16
; CHECK-32B-NEXT:    blr
entry:
  %0 = load i16*, i16** @us_addr, align 8
  %1 = bitcast i16* %0 to i8*
  %2 = call i32 @llvm.ppc.load2r(i8* %1)
  %conv = trunc i32 %2 to i16
  ret i16 %conv
}

declare i32 @llvm.ppc.load2r(i8*)

define dso_local zeroext i32 @test_builtin_ppc_load4r() {
; CHECK-64B-LABEL: test_builtin_ppc_load4r:
; CHECK-64B:         lwbrx 3, 0, 3
; CHECK-64B-NEXT:    blr

; CHECK-32B-LABEL: test_builtin_ppc_load4r:
; CHECK-32B:         lwbrx 3, 0, 3
; CHECK-32B-NEXT:    blr
entry:
  %0 = load i32*, i32** @ui_addr, align 8
  %1 = bitcast i32* %0 to i8*
  %2 = call i32 @llvm.ppc.load4r(i8* %1)
  ret i32 %2
}

declare i32 @llvm.ppc.load4r(i8*)
