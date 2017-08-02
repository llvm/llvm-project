; Test basic support for some older processors.

;RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=pwr3 | FileCheck %s
;RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=pwr4 | FileCheck %s
;RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=pwr5 | FileCheck %s
;RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=pwr5x | FileCheck %s
;RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=pwr6x | FileCheck %s

define void @foo() {
entry:
  ret void
}

; CHECK: @foo
