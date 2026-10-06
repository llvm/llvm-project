; The PSV0 RuntimeInfo version is selected from the module's validator version
; so the encoded record size matches what the DXIL validator expects. A module
; without dx.valver encodes version 0; declared validator versions select
; higher PSV versions (mirroring DXC's hlsl::GetPSVVersion mapping).

; RUN: split-file %s %t
; RUN: llc %t/no_valver.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=V0
; RUN: llc %t/valver_1_7.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=V2
; RUN: llc %t/valver_1_8.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=V3

;--- no_valver.ll
target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; No dx.valver -> PSV version 0 (24-byte RuntimeInfo).
; V0:      - Name:            PSV0
; V0-NEXT:     Size:            32
; V0:          Version:         0
; V0-NOT:      NumThreadsX:
; V0-NOT:      RuntimeInfoSize:
; V0-NOT:      EntryName:

;--- valver_1_7.ll
target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}
!0 = !{i32 1, i32 7}

; Validator version 1.7 -> PSV version 2 (48-byte RuntimeInfo, no EntryName).
; V2:      - Name:            PSV0
; V2-NEXT:     Size:            68
; V2:          Version:         2
; V2:          NumThreadsX:     1
; V2:          RuntimeInfoSize: 48
; V2-NOT:      EntryName:

;--- valver_1_8.ll
target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}
!0 = !{i32 1, i32 8}

; Validator version 1.8 -> PSV version 3 (52-byte RuntimeInfo, with EntryName).
; V3:      - Name:            PSV0
; V3-NEXT:     Size:            76
; V3:          Version:         3
; V3:          NumThreadsX:     1
; V3:          EntryName:       main
; V3:          RuntimeInfoSize: 52