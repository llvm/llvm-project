; RUN: llc --mtriple=loongarch32 -mattr=+d --relocation-model=static < %s | FileCheck %s
; RUN: llc --mtriple=loongarch32 -mattr=+d --relocation-model=pic < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 -mattr=+d --relocation-model=static < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 -mattr=+d --relocation-model=pic < %s | FileCheck %s

declare void @throw_exception()

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

; CHECK-LABEL: test1:
; CHECK: .cfi_startproc
;; PersonalityEncoding = DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4
; CHECK-NEXT: .cfi_personality 155, DW.ref.__gxx_personality_v0
;; LSDAEncoding = DW_EH_PE_pcrel | DW_EH_PE_sdata4
; CHECK-NEXT: .cfi_lsda 27, .Lexception0

define void @test1() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @throw_exception() to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void
}

; CHECK-LABEL: GCC_except_table0:
; CHECK-NEXT: .Lexception0:
; CHECK-NEXT: .byte 255 # @LPStart Encoding = omit
;; TTypeEncoding = DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4
; CHECK-NEXT: .byte 155 # @TType Encoding = indirect pcrel sdata4
; CHECK: .Lttbaseref0:
;; CallSiteEncoding = dwarf::DW_EH_PE_uleb128
; CHECK-NEXT: .byte 1                       # Call site Encoding = uleb128
; CHECK-NEXT: .uleb128 .Lcst_end0-.Lcst_begin0
; CHECK-NEXT: .Lcst_begin0:
; CHECK-NEXT: .uleb128 .Ltmp0-.Lfunc_begin0   # >> Call Site 1 <<
; CHECK-NEXT: .uleb128 .Ltmp1-.Ltmp0          #   Call between .Ltmp0 and .Ltmp1
; CHECK-NEXT: .uleb128 .Ltmp2-.Lfunc_begin0   #     jumps to .Ltmp2
; CHECK-NEXT: .byte 1                       #   On action: 1
; CHECK-NEXT: .uleb128 .Ltmp1-.Lfunc_begin0   # >> Call Site 2 <<
; CHECK-NEXT: .uleb128 .Lfunc_end0-.Ltmp1     #   Call between .Ltmp1 and .Lfunc_end0
; CHECK-NEXT: .byte 0                       #     has no landing pad
; CHECK-NEXT: .byte 0                       #   On action: cleanup
