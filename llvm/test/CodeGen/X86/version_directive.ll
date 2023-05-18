; RUN: llc -mtriple x86_64-apple-darwin15.0.0 -o - /dev/null | FileCheck %s
; RUN: llc -mtriple x86_64-apple-macosx10.11.0 -o - /dev/null | FileCheck %s
; RUN: llc -mtriple x86_64-apple-macos10.11.0 -o - /dev/null | FileCheck %s

; CHECK: .macosx_version_min 10, 11
