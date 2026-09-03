; RUN: llc -mtriple=mipsel -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=UNRESERVED
; RUN: llc -mtriple=mips64el -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=UNRESERVED
; RUN: llc -mtriple=mipsel -mattr=+reserve-gpr24 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=RESERVED
; RUN: llc -mtriple=mips64el -mattr=+reserve-gpr24 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=RESERVED

; Exhaust the allocatable GPRs so this uses $24 without the reservation.
@var = global [32 x i64] zeroinitializer

define void @foo() {
; UNRESERVED-LABEL: foo:
; UNRESERVED: $24
; RESERVED-LABEL: foo:
; RESERVED-NOT: $24
; RESERVED: .end foo
  %v = load volatile [32 x i64], ptr @var
  store volatile [32 x i64] %v, ptr @var
  ret void
}
