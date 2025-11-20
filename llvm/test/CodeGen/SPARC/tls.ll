; RUN: llc <%s -mtriple=sparc-unknown-linux   -relocation-model=static | FileCheck %s --check-prefix=v8abs
; RUN: llc <%s -mtriple=sparcv9-unknown-linux -relocation-model=static | FileCheck %s --check-prefix=v9abs
; RUN: llc <%s -mtriple=sparc-unknown-linux   -relocation-model=pic    | FileCheck %s --check-prefix=pic
; RUN: llc <%s -mtriple=sparcv9-unknown-linux -relocation-model=pic    | FileCheck %s --check-prefix=pic

; RUN: llc <%s -mtriple=sparc-unknown-linux   -relocation-model=static -filetype=obj | llvm-readobj -r --symbols - | FileCheck %s --check-prefix=v8abs-obj
; RUN: llc <%s -mtriple=sparcv9-unknown-linux -relocation-model=static -filetype=obj | llvm-readobj -r --symbols - | FileCheck %s --check-prefix=v9abs-obj
; RUN: llc <%s -mtriple=sparc-unknown-linux   -relocation-model=pic    -filetype=obj | llvm-readobj -r --symbols - | FileCheck %s --check-prefix=pic-obj
; RUN: llc <%s -mtriple=sparcv9-unknown-linux -relocation-model=pic    -filetype=obj | llvm-readobj -r --symbols - | FileCheck %s --check-prefix=pic-obj

@local_symbol = internal thread_local global i32 0
@extern_symbol = external thread_local global i32

; v8abs-LABEL:  test_tls_local
; v8abs:        sethi  %tle_hix22(local_symbol), [[R0:%[goli][0-7]]]
; v8abs:        xor    [[R0]], %tle_lox10(local_symbol), [[R1:%[goli][0-7]]]
; v8abs:        ld     [%g7+[[R1]]]

; v9abs-LABEL:  test_tls_local
; v9abs:        sethi  %tle_hix22(local_symbol), [[R0:%[goli][0-7]]]
; v9abs:        xor    [[R0]], %tle_lox10(local_symbol), [[R1:%[goli][0-7]]]
; v9abs:        ld     [%g7+[[R1]]]

; pic-LABEL:  test_tls_local
; pic:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[PC:%[goli][0-7]]]
; pic:        add    [[PC]], %o7, [[GOTBASE:%[goli][0-7]]]
; pic-DAG:    sethi  %tldm_hi22(local_symbol), [[R0:%[goli][0-7]]]
; pic-DAG:    add    [[R0]], %tldm_lo10(local_symbol), [[R1:%[goli][0-7]]]
; pic-DAG:    add    [[GOTBASE]], [[R1]], %o0, %tldm_add(local_symbol)
; pic-DAG:    call   __tls_get_addr, %tldm_call(local_symbol)
; pic-DAG:    sethi  %tldo_hix22(local_symbol), [[R2:%[goli][0-7]]]
; pic-DAG:    xor    [[R2]], %tldo_lox10(local_symbol), [[R3:%[goli][0-7]]]
; pic:        add    %o0, [[R3]], {{.+}}, %tldo_add(local_symbol)

define i32 @test_tls_local() {
entry:
  %0 = load i32, ptr @local_symbol, align 4
  %1 = add i32 %0, 1
  store i32 %1, ptr @local_symbol, align 4
  ret i32 %1
}


; v8abs-LABEL:  test_tls_extern
; v8abs:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_), %[[GOTBASE:[goli][0-7]]]
; v8abs:        sethi  %tie_hi22(extern_symbol), [[R1:%[goli][0-7]]]
; v8abs:        add    [[R1]], %tie_lo10(extern_symbol), %[[R2:[goli][0-7]]]
; v8abs:        ld     [%[[GOTBASE]]+%[[R2]]], [[R3:%[goli][0-7]]], %tie_ld(extern_symbol)
; v8abs:        add    %g7, [[R3]], %[[R4:[goli][0-7]]], %tie_add(extern_symbol)
; v8abs:        ld     [%[[R4]]]

; v9abs-LABEL:  test_tls_extern
; v9abs:        or     {{%[goli][0-7]}}, %l44(_GLOBAL_OFFSET_TABLE_), %[[GOTBASE:[goli][0-7]]]
; v9abs:        sethi  %tie_hi22(extern_symbol), [[R1:%[goli][0-7]]]
; v9abs:        add    [[R1]], %tie_lo10(extern_symbol), %[[R2:[goli][0-7]]]
; v9abs:        ldx    [%[[GOTBASE]]+%[[R2]]], [[R3:%[goli][0-7]]], %tie_ldx(extern_symbol)
; v9abs:        add    %g7, [[R3]], %[[R4:[goli][0-7]]], %tie_add(extern_symbol)
; v9abs:        ld     [%[[R4]]]

; pic-LABEL:  test_tls_extern
; pic:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[PC:%[goli][0-7]]]
; pic:        add    [[PC]], %o7, [[GOTBASE:%[goli][0-7]]]
; pic:        sethi  %tgd_hi22(extern_symbol), [[R0:%[goli][0-7]]]
; pic:        add    [[R0]], %tgd_lo10(extern_symbol), [[R1:%[goli][0-7]]]
; pic:        add    [[GOTBASE]], [[R1]], %o0, %tgd_add(extern_symbol)
; pic:        call   __tls_get_addr, %tgd_call(extern_symbol)
; pic-NEXT:   nop

define i32 @test_tls_extern() {
entry:
  %0 = load i32, ptr @extern_symbol, align 4
  %1 = add i32 %0, 1
  store i32 %1, ptr @extern_symbol, align 4
  ret i32 %1
}


; v8abs-obj: Relocations [
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_HIX22 local_symbol 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_LOX10 local_symbol 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_HI22 _GLOBAL_OFFSET_TABLE_ 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_LO10 _GLOBAL_OFFSET_TABLE_ 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_HI22 extern_symbol 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_LO10 extern_symbol 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_LD extern_symbol 0x0
; v8abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_ADD extern_symbol 0x0
; v8abs-obj: ]

; v9abs-obj: Relocations [
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_HIX22 local_symbol 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_LOX10 local_symbol 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_H44 _GLOBAL_OFFSET_TABLE_ 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_M44 _GLOBAL_OFFSET_TABLE_ 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_L44 _GLOBAL_OFFSET_TABLE_ 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_HI22 extern_symbol 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_LO10 extern_symbol 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_LDX extern_symbol 0x0
; v9abs-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_IE_ADD extern_symbol 0x0
; v9abs-obj: ]

; pic-obj: Relocations [
; pic-obj:  Section {{.*}} .rela.text {
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_PC22 _GLOBAL_OFFSET_TABLE_ 0x4
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_PC10 _GLOBAL_OFFSET_TABLE_ 0x8
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_HIX22 local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_HI22 local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_LO10 local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_ADD local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_LOX10 local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_CALL local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_ADD local_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_PC22 _GLOBAL_OFFSET_TABLE_ 0x4
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_PC10 _GLOBAL_OFFSET_TABLE_ 0x8
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_GD_HI22 extern_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_GD_LO10 extern_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_GD_ADD extern_symbol 0x0
; pic-obj:    0x{{[0-9,A-F]+}} R_SPARC_TLS_GD_CALL extern_symbol 0x0
; pic-obj: ]
; pic-obj:      Symbols [
; pic-obj:        Symbol {
; pic-obj:          Name: __tls_get_addr ({{[0-9]+}})
; pic-obj-NEXT:     Value: 0x0
; pic-obj-NEXT:     Size: 0
; pic-obj-NEXT:     Binding: Global (0x1)
; pic-obj-NEXT:     Type: None (0x0)
; pic-obj-NEXT:     Other: 0
; pic-obj-NEXT:     Section: Undefined (0x0)
; pic-obj-NEXT:   }
; pic-obj:      ]

