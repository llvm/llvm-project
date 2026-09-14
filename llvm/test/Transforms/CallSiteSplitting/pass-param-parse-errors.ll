; RUN: not opt -passes='callsite-splitting<unknown>' -disable-output %s 2>&1 | FileCheck -check-prefix=UNKNOWNERR %s
; RUN: not opt -passes='callsite-splitting<duplication-threshold=>' -disable-output %s 2>&1 | FileCheck -check-prefix=DUP-EMPTY-ERR %s
; RUN: not opt -passes='callsite-splitting<duplication-threshold=x>' -disable-output %s 2>&1 | FileCheck -check-prefix=DUP-NOTINT-ERR %s

; UNKNOWNERR: invalid CallSiteSplitting pass parameter '{{.*}}'
; DUP-EMPTY-ERR: invalid argument to CallSiteSplitting pass duplication-threshold parameter: ''
; DUP-NOTINT-ERR: invalid argument to CallSiteSplitting pass duplication-threshold parameter: 'x'
