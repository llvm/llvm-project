; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s \
; RUN:   | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff < %s \
; RUN:   | FileCheck %s

; CHECK: .file "/absolute/path/to/file"
; CHECK-SAME: ,{{.*version}}

source_filename = "/absolute/path/to/file"
