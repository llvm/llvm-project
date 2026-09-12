; RUN: not opt -passes='float2int<unknown>' -disable-output %s 2>&1 | FileCheck -check-prefix=UNKNOWNERR %s
; RUN: not opt -passes='float2int<max-integer-bw=>' -disable-output %s 2>&1 | FileCheck -check-prefix=BW-EMPTY-ERR %s
; RUN: not opt -passes='float2int<max-integer-bw=x>' -disable-output %s 2>&1 | FileCheck -check-prefix=BW-NOTINT-ERR %s

; UNKNOWNERR: invalid Float2Int pass parameter '{{.*}}'
; BW-EMPTY-ERR: invalid argument to Float2Int pass max-integer-bw parameter: ''
; BW-NOTINT-ERR: invalid argument to Float2Int pass max-integer-bw parameter: 'x'
