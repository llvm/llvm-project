; RUN: llc -o /dev/null -verify-machineinstrs -mtriple=powerpc64-unknown-gnu-linux -stop-after machine-sink %s 2>&1 | FileCheck %s --check-prefix=ISEL
; RUN: llc -o /dev/null -verify-machineinstrs -mtriple=powerpc64-unknown-gnu-linux -fast-isel -fast-isel-abort=1 -stop-after machine-sink %s 2>&1 | FileCheck %s --check-prefix=FAST-ISEL

define void @caller_meta_leaf() {
entry:
  %metadata = alloca i64, i32 3, align 8
  store i64 11, i64* %metadata
  store i64 12, i64* %metadata
  store i64 13, i64* %metadata
; ISEL:      ADJCALLSTACKDOWN 0, implicit-def
; ISEL-NEXT: STACKMAP
; ISEL-NEXT: ADJCALLSTACKUP 0, 0, implicit-def
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 4, i32 0, i64* %metadata)
; FAST-ISEL:      ADJCALLSTACKDOWN 0, implicit-def
; FAST-ISEL-NEXT: STACKMAP
; FAST-ISEL-NEXT: ADJCALLSTACKUP 0, 0, implicit-def
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
