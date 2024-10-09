; RUN: not opt -passes='scalarizer<unknown>' -disable-output %s 2>&1 | FileCheck -check-prefix=UNKNOWNERR %s
; RUN: not opt -passes='scalarizer<;>' -disable-output %s 2>&1 | FileCheck -check-prefix=UNKNOWNERR %s
; RUN: not opt -passes='scalarizer<min-bits=>' -disable-output %s 2>&1 | FileCheck -check-prefix=MINBITS-EMPTY-ERR %s
; RUN: not opt -passes='scalarizer<min-bits=x>' -disable-output %s 2>&1 | FileCheck -check-prefix=MINBITS-NOTINT-ERR %s
; RUN: not opt -passes='scalarizer<no-min-bits=10>' -disable-output %s 2>&1 | FileCheck -check-prefix=UNKNOWNERR %s

; UNKNOWNERR: invalid Scalarizer pass parameter '{{.*}}'
; MINBITS-EMPTY-ERR: invalid argument to Scalarizer pass min-bits parameter: ''
; MINBITS-NOTINT-ERR: invalid argument to Scalarizer pass min-bits parameter: 'x'
