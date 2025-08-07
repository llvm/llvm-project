; RUN: split-file %s %t
; RUN: not llvm-as < %t/missing-args.ll 2>&1 | FileCheck %s --check-prefix=MISSING-ARGS
; RUN: not llvm-as < %t/empty.ll 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not llvm-as < %t/unterminated.ll 2>&1 | FileCheck %s --check-prefix=UNTERMINATED
; RUN: not llvm-as < %t/invalid-kind.ll 2>&1 | FileCheck %s --check-prefix=INVALID-KIND
; RUN: not llvm-as < %t/other.ll 2>&1 | FileCheck %s --check-prefix=OTHER
; RUN: not llvm-as < %t/missing-colon.ll 2>&1 | FileCheck %s --check-prefix=MISSING-COLON
; RUN: not llvm-as < %t/invalid-access-kind.ll 2>&1 | FileCheck %s --check-prefix=INVALID-ACCESS-KIND
; RUN: not llvm-as < %t/default-after-loc.ll 2>&1 | FileCheck %s --check-prefix=DEFAULT-AFTER-LOC

;--- missing-args.ll
; MISSING-ARGS: error: expected '('
declare void @fn() memory
;--- empty.ll
; EMPTY: error: expected memory location (argmem, inaccessiblemem, errnomem) or access kind (none, read, write, readwrite)
declare void @fn() memory()
;--- unterminated.ll
; UNTERMINATED: error: unterminated memory attribute
declare void @fn() memory(read
;--- invalid-kind.ll
; INVALID-KIND: error: expected memory location (argmem, inaccessiblemem, errnomem) or access kind (none, read, write, readwrite)
declare void @fn() memory(foo)
;--- other.ll
; OTHER: error: expected memory location (argmem, inaccessiblemem, errnomem) or access kind (none, read, write, readwrite)
declare void @fn() memory(other: read)
;--- missing-colon.ll
; MISSING-COLON: error: expected ':' after location
declare void @fn() memory(argmem)
;--- invalid-access-kind.ll
; INVALID-ACCESS-KIND: error: expected access kind (none, read, write, readwrite)
declare void @fn() memory(argmem: foo)
;--- default-after-loc.ll
; DEFAULT-AFTER-LOC: error: default access kind must be specified first
declare void @fn() memory(argmem: read, write)
