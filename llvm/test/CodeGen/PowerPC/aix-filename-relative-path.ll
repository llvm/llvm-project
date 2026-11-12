; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s \
; RUN:   | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff < %s \
; RUN:   | FileCheck %s

; CHECK: .file "../relative/path/to/file"
; CHECK-SAME: ,{{.*version}}

source_filename = "../relative/path/to/file"
