; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff  < %s | FileCheck %s

; CHECK:      .file "1.c"
; CHECK-NEXT: .csect ..text..[PR],5
; CHECK-NEXT: .rename ..text..[PR],""
; CHECK-NEXT: .machine "PWR8"

source_filename = "1.c"

define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}

attributes #0 = {"target-cpu"="pwr8"}
