; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s \
; RUN:   | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff < %s \
; RUN:   | FileCheck %s

; CHECK: .file "1""2.c"

source_filename = "1\222.c"
