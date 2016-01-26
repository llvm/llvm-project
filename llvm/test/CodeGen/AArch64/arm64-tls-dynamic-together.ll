; RUN: llc -O0 -mtriple=arm64-none-linux-gnu -relocation-model=pic \
; RUN:     -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK -check-prefix=NOEMU %s
; RUN: llc -emulated-tls -O0 -mtriple=arm64-none-linux-gnu -relocation-model=pic \
; RUN:     -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK -check-prefix=EMU %s

; If the .tlsdesccall and blr parts are emitted completely separately (even with
; glue) then LLVM will separate them quite happily (with a spill at O0, hence
; the option). This is definitely wrong, so we make sure they are emitted
; together.

@general_dynamic_var = external thread_local global i32

define i32 @test_generaldynamic() {
; CHECK-LABEL: test_generaldynamic:

  %val = load i32, i32* @general_dynamic_var
  ret i32 %val

; NOEMU: .tlsdesccall general_dynamic_var
; NOEMU-NEXT: blr {{x[0-9]+}}
; NOEMU-NOT: __emutls_v.general_dynamic_var:

; EMU: adrp{{.+}}__emutls_v.general_dynamic_var
; EMU: bl __emutls_get_address

; EMU-NOT: __emutls_v.general_dynamic_var
; EMU-NOT: __emutls_t.general_dynamic_var
}

@emulated_init_var = thread_local global i32 37, align 8

define i32 @test_emulated_init() {
; COMMON-LABEL: test_emulated_init:

  %val = load i32, i32* @emulated_init_var
  ret i32 %val

; EMU: adrp{{.+}}__emutls_v.emulated_init_var
; EMU: bl __emutls_get_address

; EMU-NOT: __emutls_v.general_dynamic_var:

; EMU:      .p2align 3
; EMU-LABEL: __emutls_v.emulated_init_var:
; EMU-NEXT: .xword 4
; EMU-NEXT: .xword 8
; EMU-NEXT: .xword 0
; EMU-NEXT: .xword __emutls_t.emulated_init_var

; EMU-LABEL: __emutls_t.emulated_init_var:
; EMU-NEXT: .word 37
}

; CHECK-NOT: __emutls_v.general_dynamic_var:
; EMU-NOT: __emutls_t.general_dynamic_var
