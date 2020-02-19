; Verify that we use the vector ABI datalayout if and only if
; the vector facility is present.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=generic | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s

; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=vector | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=+vector | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=-vector,vector | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=-vector,+vector | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=-vector | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=vector,-vector | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=+vector,-vector | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s

; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=-vector | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s

; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=+soft-float | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   -mattr=soft-float,-soft-float | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   -mattr=-soft-float,soft-float | \
; RUN:   FileCheck -check-prefix=CHECK-NOVECTOR %s

%struct.S = type { i8, <2 x i64> }

define void @test(%struct.S* %s) nounwind {
; CHECK-VECTOR-LABEL: @test
; CHECK-VECTOR: vl %v0, 8(%r2)
; CHECK-NOVECTOR-LABEL: @test
; CHECK-NOVECTOR-DAG: agsi 16(%r2), 1
; CHECK-NOVECTOR-DAG: agsi 24(%r2), 1
  %ptr = getelementptr %struct.S, %struct.S* %s, i64 0, i32 1
  %vec = load <2 x i64>, <2 x i64>* %ptr
  %add = add <2 x i64> %vec, <i64 1, i64 1>
  store <2 x i64> %add, <2 x i64>* %ptr
  ret void
}

