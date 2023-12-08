; Test (fast) serialization.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s --check-prefix=Z10
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196  | FileCheck %s --check-prefix=Z196
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s --check-prefix=ZEC12
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13   | FileCheck %s --check-prefix=Z13

define void @test() {
; Z10:   bcr 15, %r0
; Z196:  bcr 14, %r0
; ZEC12: bcr 14, %r0
; Z13:   bcr 14, %r0
  fence seq_cst
  ret void
}

