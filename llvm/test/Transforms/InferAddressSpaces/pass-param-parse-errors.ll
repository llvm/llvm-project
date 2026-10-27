; RUN: not opt -passes='infer-address-spaces<unknown>' -disable-output %s 2>&1 | FileCheck -check-prefix=UNKNOWNERR %s

; UNKNOWNERR: invalid InferAddressSpacesPass pass parameter '{{.*}}'
