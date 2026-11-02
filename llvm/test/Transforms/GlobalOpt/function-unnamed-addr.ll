; RUN: opt -passes=globalopt -S < %s | FileCheck %s
;
; Skipping use analysis for functions whose unnamed_addr cannot be strengthened
; must still allow local functions to be upgraded and other optimizations to run.

@mutable = internal unnamed_addr global i32 9

; CHECK-LABEL: define internal fastcc void @internal_local() unnamed_addr {
define internal void @internal_local() local_unnamed_addr {
  ret void
}

; CHECK-LABEL: define private fastcc void @private_local() unnamed_addr {
define private void @private_local() local_unnamed_addr {
  ret void
}

; Address comparisons prevent strengthening local_unnamed_addr.
; CHECK-LABEL: define internal void @compared_local() local_unnamed_addr {
define internal void @compared_local() local_unnamed_addr {
  ret void
}

; Saturated functions must still be eligible for calling-convention changes.
; CHECK-LABEL: define internal fastcc void @internal_global() unnamed_addr {
define internal void @internal_global() unnamed_addr {
  ret void
}

; CHECK-LABEL: define internal fastcc void @internal_plain() unnamed_addr {
define internal void @internal_plain() {
  ret void
}

; CHECK-LABEL: define void @external_plain() local_unnamed_addr {
define void @external_plain() {
  ret void
}

; CHECK-LABEL: define void @external_local() local_unnamed_addr {
define void @external_local() local_unnamed_addr {
  ret void
}

; CHECK-LABEL: define void @external_global() unnamed_addr {
define void @external_global() unnamed_addr {
  ret void
}

; CHECK: declare void @decl_local() local_unnamed_addr
declare void @decl_local() local_unnamed_addr

; CHECK-LABEL: define i1 @compare(
; CHECK: icmp eq ptr @compared_local, %p
define i1 @compare(ptr %p) {
  %v = icmp eq ptr @compared_local, %p
  ret i1 %v
}

; CHECK-LABEL: define void @caller()
; CHECK: call fastcc void @internal_local()
; CHECK-NEXT: call fastcc void @private_local()
; CHECK-NEXT: call fastcc void @internal_global()
; CHECK-NEXT: call fastcc void @internal_plain()
define void @caller() {
  call void @internal_local()
  call void @private_local()
  call void @internal_global()
  call void @internal_plain()
  call void @external_plain()
  call void @external_local()
  call void @external_global()
  call void @decl_local()
  ret void
}

; Saturated global variables must still be optimized.
; CHECK-LABEL: define i32 @read_mutable()
; CHECK-NEXT: ret i32 9
define i32 @read_mutable() {
  %v = load i32, ptr @mutable
  ret i32 %v
}
