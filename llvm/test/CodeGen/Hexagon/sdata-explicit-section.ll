; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s
; RUN: llc -march=hexagon -relocation-model=pic < %s | FileCheck %s

; CHECK: .section .sdata.4,"aws",@progbits
@g0 = global i32 zeroinitializer, section ".sdata"

