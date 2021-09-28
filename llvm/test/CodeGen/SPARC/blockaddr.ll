; RUN: llc < %s -mtriple=sparc   -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=medium | FileCheck --check-prefix=abs44 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=large  | FileCheck --check-prefix=abs64 %s
; RUN: llc < %s -mtriple=sparc   -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v8pic32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v9pic32 %s

;
; copied from test/CodeGen/Mips/blockaddr.ll and modified for SPARC
;
@reg = common global i8* null, align 4

define i8* @dummy(i8* %x) nounwind readnone noinline {
entry:
  ret i8* %x
}

; abs32-LABEL: func_block_addr:
; abs32: sethi %hi([[BLK:.+]]), [[R:%[gilo][0-7]]]
; abs32: call dummy
; abs32: add  [[R]], %lo([[BLK]]), %o0
; abs32: jmp %o0

; abs44-LABEL: func_block_addr:
; abs44: sethi %h44([[BLK:.+]]), [[R:%[gilo][0-7]]]
; abs44: add [[R]], %m44([[BLK]]), [[R1:%[gilo][0-7]]]
; abs44: sllx [[R1]], 12, [[R2:%[gilo][0-7]]]
; abs44: call dummy
; abs44: add [[R2]], %l44([[BLK]]), %o0
; abs44: jmp %o0

; abs64-LABEL: func_block_addr:
; abs64: sethi %hi([[BLK:.+]]), [[R:%[gilo][0-7]]]
; abs64: add [[R]], %lo([[BLK]]), [[R1:%[gilo][0-7]]]
; abs64: sethi %hh([[BLK]]), [[R2:%[gilo][0-7]]]
; abs64: add [[R2]], %hm([[BLK]]), [[R3:%[gilo][0-7]]]
; abs64: sllx [[R3]], 32,  [[R4:%[gilo][0-7]]]
; abs64: call dummy
; abs64: add [[R2]], [[R1]], %o0
; abs64: jmp %o0


; v8pic32: func_block_addr
; v8pic32: sethi %hi(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R:%[gilo][0-7]]]
; v8pic32: or [[R]], %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R1:%[gilo][0-7]]]
; v8pic32: add [[R1]], %o7, %[[R2:[gilo][0-7]]]
; v8pic32: sethi %hi([[BLK:.+]]), [[R3:%[gilo][0-7]]]
; v8pic32: add  [[R3]], %lo([[BLK]]), %[[R4:[gilo][0-7]]]
; v8pic32: call dummy
; v8pic32: ld [%[[R2]]+%[[R4]]], %o0
; v8pic32: jmp %o0


; v9pic32: func_block_addr
; v9pic32: sethi %hi(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R:%[gilo][0-7]]]
; v9pic32: or [[R]], %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R1:%[gilo][0-7]]]
; v9pic32: add [[R1]], %o7, %[[R2:[gilo][0-7]]]
; v9pic32: sethi %hi([[BLK:.+]]), [[R3:%[gilo][0-7]]]
; v9pic32: add  [[R3]], %lo([[BLK]]), %[[R4:[gilo][0-7]]]
; v9pic32: call dummy
; v9pic32: ldx [%[[R2]]+%[[R4]]], %o0
; v9pic32: jmp %o0


define void @func_block_addr() nounwind {
entry:
  %call = tail call i8* @dummy(i8* blockaddress(@func_block_addr, %baz))
  indirectbr i8* %call, [label %baz, label %foo]

foo:                                              ; preds = %foo, %entry
  store i8* blockaddress(@func_block_addr, %foo), i8** @reg, align 4
  br label %foo

baz:                                              ; preds = %entry
  store i8* null, i8** @reg, align 4
  ret void
}
