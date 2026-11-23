; Test that on AArch64 the personality, LSDA, and TType EH encodings already
; default to 8-byte (sdata8), so --large-eh-encoding leaves them unchanged.
; The flag's only effect on AArch64 is the FDE CFI encoding, which is checked
; in llvm/test/MC/AArch64/large-eh-encoding.s.

; Without the flag, with the flag, and with the flag under PIC all produce the
; same sdata8 encodings (the AArch64 case does not depend on the reloc model).
; RUN: llc -mtriple aarch64-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple aarch64-linux-gnu --large-eh-encoding < %s | FileCheck %s
; RUN: llc -mtriple aarch64-linux-gnu -relocation-model=pic --large-eh-encoding < %s | FileCheck %s

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

;; AArch64 always uses these regardless of --large-eh-encoding / reloc model:
;; PersonalityEncoding = DW_EH_PE_indirect|DW_EH_PE_pcrel|DW_EH_PE_sdata8 = 156
;; LSDAEncoding        = DW_EH_PE_pcrel|DW_EH_PE_sdata8 = 28
; CHECK: .cfi_personality 156, DW.ref.__gxx_personality_v0
; CHECK: .cfi_lsda 28,
