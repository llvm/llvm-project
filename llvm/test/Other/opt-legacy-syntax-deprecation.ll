; REQUIRES: x86-registered-target

; RUN: opt /dev/null -disable-output 2>&1 | FileCheck %s --check-prefix=OK --allow-empty
; RUN: opt /dev/null -disable-output -passes=instcombine 2>&1 | FileCheck %s --check-prefix=OK --allow-empty
; RUN: not opt /dev/null -disable-output -instcombine 2>&1 | FileCheck %s --check-prefixes=UNKNOWN,WARN
; RUN: not opt /dev/null -disable-output -bogus 2>&1 | FileCheck %s --check-prefix=BOGUS --implicit-check-not='opt -passname'
; RUN: not opt /dev/null -disable-output -passes=instcombine -codegenprepare -mtriple=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s --check-prefix=WARN
; RUN: opt /dev/null -disable-output -codegenprepare -mtriple=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s --check-prefix=OK --allow-empty

; OK-NOT: deprecated

; UNKNOWN: Unknown command line argument '-instcombine'

; BOGUS: Unknown command line argument '-bogus'

; WARN: The `opt -passname` syntax for the new pass manager is not supported, please use `opt -passes=<pipeline>` (or the `-p` alias for a more concise version).
