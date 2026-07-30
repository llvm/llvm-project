; RUN: llc < %s -mtriple=sparc   -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=medium | FileCheck --check-prefix=abs44 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=large  | FileCheck --check-prefix=abs64 %s
; RUN: llc < %s -mtriple=sparc   -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v8pic32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v9pic32 %s

@G = external global i8

define zeroext i8 @loadG() {
  %tmp = load i8, ptr @G
  ret i8 %tmp
}

; abs32: loadG
; abs32: sethi %hi(G), %[[R:[gilo][0-7]]]
; abs32: retl
; abs32: ldub [%[[R]]+%lo(G)], %o0


; abs44: loadG
; abs44: sethi %h44(G), %[[R1:[gilo][0-7]]]
; abs44: add %[[R1]], %m44(G), %[[R2:[gilo][0-7]]]
; abs44: sllx %[[R2]], 12, %[[R3:[gilo][0-7]]]
; abs44: retl
; abs44: ldub [%[[R3]]+%l44(G)], %o0


; abs64: loadG
; abs64: sethi %hi(G), %[[R1:[gilo][0-7]]]
; abs64: add %[[R1]], %lo(G), %[[R2:[gilo][0-7]]]
; abs64: sethi %hh(G), %[[R3:[gilo][0-7]]]
; abs64: add %[[R3]], %hm(G), %[[R4:[gilo][0-7]]]
; abs64: sllx %[[R4]], 32, %[[R5:[gilo][0-7]]]
; abs64: retl
; abs64: ldub [%[[R5]]+%[[R2]]], %o0


; v8pic32: loadG
; v8pic32: _GLOBAL_OFFSET_TABLE_
; v8pic32: sethi %hi(G), %[[R1:[gilo][0-7]]]
; v8pic32: add %[[R1]], %lo(G), %[[Goffs:[gilo][0-7]]]
; v8pic32: ld [%[[GOT:[gilo][0-7]]]+%[[Goffs]]], %[[Gaddr:[gilo][0-7]]]
; v8pic32: ldub [%[[Gaddr]]], %i0
; v8pic32: ret
; v8pic32: restore


; v9pic32: loadG
; v9pic32: _GLOBAL_OFFSET_TABLE_
; v9pic32: sethi %hi(G), %[[R1:[gilo][0-7]]]
; v9pic32: add %[[R1]], %lo(G), %[[Goffs:[gilo][0-7]]]
; v9pic32: ldx [%[[GOT:[gilo][0-7]]]+%[[Goffs]]], %[[Gaddr:[gilo][0-7]]]
; v9pic32: ldub [%[[Gaddr]]], %i0
; v9pic32: ret
; v9pic32: restore

