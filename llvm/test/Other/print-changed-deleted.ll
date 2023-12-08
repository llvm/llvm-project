; RUN: opt -passes=globaldce < %s -disable-output -print-changed -filter-print-funcs=f 2>&1 | FileCheck %s

; CHECK-NOT: IR Dump After GlobalDCEPass
; CHECK: IR Deleted After GlobalDCEPass
; CHECK-NOT: IR Dump After GlobalDCEPass

declare void @f()
