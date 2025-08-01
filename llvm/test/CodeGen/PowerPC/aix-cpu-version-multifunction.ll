; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff  < %s | FileCheck %s

; For the .machine directive emitted on AIX, the "target-cpu" attribute that is
; the newest will be used as the CPU for the module (in this case, PWR10).

; CHECK:      .file "file.c"
; CHECK-NEXT: .csect ..text..[PR],5
; CHECK-NEXT: .rename ..text..[PR],""
; CHECK-NEXT: .machine "PWR10"
; CHECK-NOT:  .machine "PWR8"

source_filename = "file.c"

define dso_local signext i32 @testFunc1() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}

define dso_local signext i32 @testFunc2() #1 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}

attributes #0 = { "target-cpu" = "pwr8" }
attributes #1 = { "target-cpu" = "pwr10" }

