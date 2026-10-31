; RUN: split-file %s %t
; RUN: not llvm-as %t/bad-field.ll                -disable-output 2>&1 | FileCheck %s --check-prefix BAD-FIELD
; RUN: not llvm-as %t/invalid-nodebug.ll          -disable-output 2>&1 | FileCheck %s --check-prefix INVALID-NODEBUG
; RUN: not llvm-as %t/non-array-storage.ll        -disable-output 2>&1 | FileCheck %s --check-prefix NON-ARRAY-STORAGE
; RUN: not llvm-as %t/duplicate-storage-arg.ll    -disable-output 2>&1 | FileCheck %s --check-prefix DUPLICATE-STORAGE-ARG
; RUN: not llvm-as %t/trailing-context-comma.ll   -disable-output 2>&1 | FileCheck %s --check-prefix TRAILING-CONTEXT-COMMA
; RUN: not llvm-as %t/empty-storage-comma.ll      -disable-output 2>&1 | FileCheck %s --check-prefix EMPTY-STORAGE-COMMA
; RUN: not llvm-as %t/non-flmd-storage-type.ll    -disable-output 2>&1 | FileCheck %s --check-prefix NON-FLMD-STORAGE-TYPE
; RUN: not llvm-as %t/wrong-flmd-storage-type.ll  -disable-output 2>&1 | FileCheck %s --check-prefix WRONG-FLMD-STORAGE-TYPE
; RUN: not llvm-as %t/bad-storage-field.ll        -disable-output 2>&1 | FileCheck %s --check-prefix BAD-STORAGE-FIELD
; RUN: not llvm-as %t/non-mdnode-type.ll          -disable-output 2>&1 | FileCheck %s --check-prefix NON-MDNODE-TYPE

; RUN: llvm-as %t/trailing-storage-comma.ll       -disable-output 2>&1
; RUN: llvm-as %t/no-fields.ll                    -disable-output 2>&1
; RUN: llvm-as %t/all-fields.ll                   -disable-output 2>&1
; RUN: llvm-as %t/all-fields-order.ll             -disable-output 2>&1

;; Testing invalid values.
;--- bad-field.ll
; BAD-FIELD: :2:39: error: invalid field 'madeUpField'
!0 = distinct !MDFunctionLocalContext(madeUpField: true)
;--- invalid-nodebug.ll
; INVALID-NODEBUG: :2:49: error: expected 'true' or 'false'
!0 = distinct !MDFunctionLocalContext(nonDebug: 2)
;--- non-array-storage.ll
; NON-ARRAY-STORAGE: :2:47: error: expected '[' at FLMD storage field
!0 = distinct !MDFunctionLocalContext(scopes: 5)
;--- duplicate-storage-arg.ll
; DUPLICATE-STORAGE-ARG: :2:51: error: field 'scopes' cannot be specified more than once
!0 = distinct !MDFunctionLocalContext(scopes: [], scopes: [])
;--- trailing-context-comma.ll
; TRAILING-CONTEXT-COMMA: :2:50: error: expected field label here
!0 = distinct !MDFunctionLocalContext(scopes: [],)
;--- empty-storage-comma.ll
; EMPTY-STORAGE-COMMA: :2:48: error: expected !!scope
!0 = distinct !MDFunctionLocalContext(scopes: [,])
;--- non-flmd-storage-type.ll
; NON-FLMD-STORAGE-TYPE: :2:48: error: expected !!scope
!0 = distinct !MDFunctionLocalContext(scopes: [i32 0])
;--- wrong-flmd-storage-type.ll
; WRONG-FLMD-STORAGE-TYPE: :2:48: error: expected !!scope
!0 = distinct !MDFunctionLocalContext(scopes: [!!srcLoc(line: 4, column: 1, scope: 0)])
;--- bad-storage-field.ll
; BAD-STORAGE-FIELD: :2:56: error: invalid field 'badField'
!0 = distinct !MDFunctionLocalContext(scopes: [!!scope(badField: true)])
;--- non-mdnode-type.ll
; NON-MDNODE-TYPE: :2:63: error: expected '!' here
!0 = distinct !MDFunctionLocalContext(scopes: [!!scope(scope: i32 0)])
!0 = distinct !MDFunctionLocalContext(inlinedCalls: [!!inlinedCall(inlineeContext: !"notANode")])

;; Testing valid values.
;--- trailing-storage-comma.ll
!0 = distinct !MDFunctionLocalContext(scopes: [!!scope(scope: !1),])
!1 = !{}
;--- no-fields.ll
!0 = distinct !MDFunctionLocalContext()
;--- all-fields.ll
!0 = distinct !MDFunctionLocalContext(nonDebug: true, scopes: [], srcLocs: [], inlinedCalls: [])
;--- all-fields-order.ll
!0 = distinct !MDFunctionLocalContext(inlinedCalls: [], srcLocs: [], scopes: [], nonDebug: true)
