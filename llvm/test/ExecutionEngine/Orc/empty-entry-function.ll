; RUN: not lli --entry-function= %s 2>&1 | FileCheck %s
;
; Test empty --entry-function yields an error.
; CHECK: error: --entry-function name cannot be empty
