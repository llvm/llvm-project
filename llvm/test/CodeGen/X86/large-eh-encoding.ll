; Test that --large-eh-encoding uses 8-byte encodings for personality, LSDA,
; and TType in x86_64 ELF targets.

; Default (small code model, no flag): sdata4 encodings
; RUN: llc -mtriple x86_64-pc-linux-gnu -code-model=small < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple x86_64-pc-linux-gnu -code-model=small -relocation-model=pic < %s | FileCheck %s --check-prefix=DEFAULT-PIC

; With --large-eh-encoding: sdata8 encodings
; RUN: llc -mtriple x86_64-pc-linux-gnu -code-model=small --large-eh-encoding < %s | FileCheck %s --check-prefix=LARGE
; RUN: llc -mtriple x86_64-pc-linux-gnu -code-model=small -relocation-model=pic --large-eh-encoding < %s | FileCheck %s --check-prefix=LARGE-PIC

@_ZTIi = external constant ptr

define i32 @main() uwtable personality ptr @__gxx_personality_v0 {
entry:
  invoke void @foo()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %try.cont

try.cont:
  ret i32 0
}

declare void @foo()
declare i32 @__gxx_personality_v0(...)

;; Non-PIC, default: PersonalityEncoding = DW_EH_PE_udata4 = 3
;; Non-PIC, default: LSDAEncoding = DW_EH_PE_udata4 = 3
; DEFAULT:      .cfi_personality 3, __gxx_personality_v0
; DEFAULT-NEXT: .cfi_lsda 3,

;; PIC, default (small): PersonalityEncoding = DW_EH_PE_indirect|DW_EH_PE_pcrel|DW_EH_PE_sdata4 = 155
;; PIC, default (small): LSDAEncoding = DW_EH_PE_pcrel|DW_EH_PE_sdata4 = 27
; DEFAULT-PIC:      .cfi_personality 155, DW.ref.__gxx_personality_v0
; DEFAULT-PIC-NEXT: .cfi_lsda 27,

;; Non-PIC, large-eh-encoding: PersonalityEncoding = DW_EH_PE_absptr = 0
;; Non-PIC, large-eh-encoding: LSDAEncoding = DW_EH_PE_absptr = 0
; LARGE:      .cfi_personality 0, __gxx_personality_v0
; LARGE-NEXT: .cfi_lsda 0,

;; PIC, large-eh-encoding: PersonalityEncoding = DW_EH_PE_indirect|DW_EH_PE_pcrel|DW_EH_PE_sdata8 = 156
;; PIC, large-eh-encoding: LSDAEncoding = DW_EH_PE_pcrel|DW_EH_PE_sdata8 = 28
; LARGE-PIC:      .cfi_personality 156, DW.ref.__gxx_personality_v0
; LARGE-PIC-NEXT: .cfi_lsda 28,
