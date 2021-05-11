; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnu -code-model=small < %s | FileCheck %s --check-prefixes=CHECK,CHECK-SMALL,CHECK-NON-PIC,CHECK-NON-PIC-SMALL,CHECK-NON-PIC-X64
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnux32 -code-model=small < %s | FileCheck %s --check-prefixes=CHECK,CHECK-SMALL,CHECK-NON-PIC,CHECK-NON-PIC-SMALL,CHECK-NON-PIC-X32
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnu -code-model=medium < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MEDIUM,CHECK-NON-PIC,CHECK-NON-PIC-MEDIUM,CHECK-NON-PIC-X64
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnu -code-model=large < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NON-PIC,CHECK-NON-PIC-LARGE,CHECK-NON-PIC-X64
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnu -relocation-model=pic -code-model=small < %s | FileCheck %s --check-prefixes=CHECK,CHECK-SMALL,CHECK-PIC,CHECK-PIC-SMALL,CHECK-PIC-X64
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnux32 -relocation-model=pic -code-model=small < %s | FileCheck %s --check-prefixes=CHECK,CHECK-SMALL,CHECK-PIC,CHECK-PIC-SMALL,CHECK-PIC-X32
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnu -relocation-model=pic -code-model=medium < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MEDIUM,CHECK-PIC,CHECK-PIC-MEDIUM,CHECK-PIC-X64
; RUN: llc -basic-block-sections=all -mtriple x86_64-pc-linux-gnu -relocation-model=pic -code-model=large < %s | FileCheck %s --check-prefixes=CHECK,CHECK-PIC,CHECK-PIC-LARGE,CHECK-PIC-X64
@_ZTIi = external constant i8*

define i32 @main() uwtable optsize ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; Verify that each basic block section gets its own LSDA exception symbol.
;
; CHECK-LABEL:        main:
; CHECK-NEXT:         .Lfunc_begin0:
; CHECK-NEXT:           .cfi_startproc

;; Verify personality function and LSDA encoding for NON-PIC mode.
; PersonalityEncoding = dwarf::DW_EH_PE_udata4 (small/medium)
; PersonalityEncoding = dwarf::DW_EH_PE_absptr (large)
; CHECK-NON-PIC-SMALL-NEXT:  .cfi_personality 3, __gxx_personality_v0
; CHECK-NON-PIC-MEDIUM-NEXT: .cfi_personality 3, __gxx_personality_v0
; CHECK-NON-PIC-LARGE-NEXT:  .cfi_personality 0, __gxx_personality_v0
; LSDAEncoding = dwarf::DW_EH_PE_udata4 (small)
; LSDAEncoding = dwarf::DW_EH_PE_absptr (medium/large)
; CHECK-NON-PIC-SMALL-NEXT:  .cfi_lsda 3, .Lexception0
; CHECK-NON-PIC-MEDIUM-NEXT: .cfi_lsda 0, .Lexception0
; CHECK-NON-PIC-LARGE-NEXT:  .cfi_lsda 0, .Lexception0

;; Verify personality function and LSDA encoding for PIC mode.
; PersonalityEncoding = DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4 (small/medium)
; PersonalityEncoding = DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata8 (large)
; CHECK-PIC-SMALL-NEXT:  .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-PIC-MEDIUM-NEXT: .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-PIC-LARGE-NEXT:  .cfi_personality 156, DW.ref.__gxx_personality_v0
; LSDAEncoding = DW_EH_PE_pcrel | DW_EH_PE_sdata4 (small)
; LSDAEncoding = DW_EH_PE_pcrel | DW_EH_PE_sdata8 (medium/large)
; CHECK-PIC-SMALL-NEXT:  .cfi_lsda 27, .Lexception0
; CHECK-PIC-MEDIUM-NEXT: .cfi_lsda 28, .Lexception0
; CHECK-PIC-LARGE-NEXT:  .cfi_lsda 28, .Lexception0

; CHECK-LABEL:        .Ltmp0:
; CHECK-SMALL-NEXT:     callq   _Z1fv
; CHECK-MEDIUM-NEXT:    callq   _Z1fv
; CHECK-NON-PIC-LARGE-NEXT: movabsq $_Z1fv, %rax
; CHECK-NON-PIC-LARGE-NEXT: callq   *%rax
; CHECK-PIC-LARGE-NEXT:     movabsq $_Z1fv@GOT, %rax
; CHECK-PIC-LARGE-NEXT:     callq *(%rbx,%rax)
; CHECK-LABEL:        .Ltmp1:

; CHECK-NOT:            .cfi_lsda

; CHECK-LABEL:        main.__part.1:
; CHECK-NEXT:           .cfi_startproc

; CHECK-NON-PIC-SMALL-NEXT:  .cfi_personality 3, __gxx_personality_v0
; CHECK-NON-PIC-MEDIUM-NEXT: .cfi_personality 3, __gxx_personality_v0
; CHECK-NON-PIC-LARGE-NEXT:  .cfi_personality 0, __gxx_personality_v0
; CHECK-NON-PIC-SMALL-NEXT:  .cfi_lsda 3, .Lexception1
; CHECK-NON-PIC-MEDIUM-NEXT: .cfi_lsda 0, .Lexception1
; CHECK-NON-PIC-LARGE-NEXT:  .cfi_lsda 0, .Lexception1

; CHECK-PIC-SMALL-NEXT:  .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-PIC-MEDIUM-NEXT: .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-PIC-LARGE-NEXT:  .cfi_personality 156, DW.ref.__gxx_personality_v0
; CHECK-PIC-SMALL-NEXT:  .cfi_lsda 27, .Lexception1
; CHECK-PIC-MEDIUM-NEXT: .cfi_lsda 28, .Lexception1
; CHECK-PIC-LARGE-NEXT:  .cfi_lsda 28, .Lexception1

; CHECK-NOT:            .cfi_lsda

; CHECK-LABEL:        main.__part.2:
; CHECK-NEXT:           .cfi_startproc

; CHECK-NON-PIC-SMALL-NEXT:  .cfi_personality 3, __gxx_personality_v0
; CHECK-NON-PIC-MEDIUM-NEXT: .cfi_personality 3, __gxx_personality_v0
; CHECK-NON-PIC-LARGE-NEXT:  .cfi_personality 0, __gxx_personality_v0
; CHECK-NON-PIC-SMALL-NEXT:  .cfi_lsda 3, .Lexception2
; CHECK-NON-PIC-MEDIUM-NEXT: .cfi_lsda 0, .Lexception2
; CHECK-NON-PIC-LARGE-NEXT:  .cfi_lsda 0, .Lexception2

; CHECK-PIC-SMALL-NEXT:  .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-PIC-MEDIUM-NEXT: .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-PIC-LARGE-NEXT:  .cfi_personality 156, DW.ref.__gxx_personality_v0
; CHECK-PIC-SMALL-NEXT:  .cfi_lsda 27, .Lexception2
; CHECK-PIC-MEDIUM-NEXT: .cfi_lsda 28, .Lexception2
; CHECK-PIC-LARGE-NEXT:  .cfi_lsda 28, .Lexception2

; CHECK:                nop
; CHECK-LABEL:        .Ltmp2:
; CHECK-LABEL:        .LBB_END0_2:

; CHECK-NOT:            .cfi_lsda

entry:
  invoke void @_Z1fv() optsize
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume

try.cont:
  ret i32 0

eh.resume:
  resume { i8*, i32 } %0
}

declare void @_Z1fv() optsize

declare i32 @__gxx_personality_v0(...)
;; Verify that the exception table gets split across the three basic block sections.
;
; CHECK:                .section .gcc_except_table
; CHECK-NEXT:           .p2align 2
; CHECK-NEXT:         GCC_except_table0:
; CHECK-NEXT:         .Lexception0:

;; Verify @LPStart encoding for NON-PIC mode.
; CHECK-NON-PIC-NEXT:   .byte	0                       # @LPStart Encoding = absptr
; CHECK-NON-PIC-X64-NEXT: .quad	main.__part.2
; CHECK-NON-PIC-X32-NEXT: .long	main.__part.2

;; Verify @LPStart encoding for PIC mode.
; CHECK-PIC-NEXT:       .byte	16                      # @LPStart Encoding = pcrel
; CHECK-PIC-NEXT:     [[DOT:\.Ltmp[0-9]+]]:
; CHECK-PIC-X64-NEXT:   .quad	main.__part.2-[[DOT]]
; CHECK-PIC-X32-NEXT:   .long	main.__part.2-[[DOT]]

;; Verify @TType encoding for NON-PIC mode.
; CHECK-NON-PIC-SMALL-NEXT:  .byte	3               # @TType Encoding = udata4
; CHECK-NON-PIC-MEDIUM-NEXT: .byte	0               # @TType Encoding = absptr
; CHECK-NON-PIC-LARGE-NEXT:  .byte	0               # @TType Encoding = absptr

;; Verify @TType encoding for PIC mode.
; CHECK-PIC-SMALL-NEXT: .byte 155                       # @TType Encoding = indirect pcrel sdata4
; CHECK-PIC-MEDIUM-NEXT:.byte 155                       # @TType Encoding = indirect pcrel sdata4
; CHECK-PIC-LARGE-NEXT: .byte 156                       # @TType Encoding = indirect pcrel sdata8

; CHECK-NEXT:           .uleb128 .Lttbase0-.Lttbaseref0
; CHECK-NEXT:         .Lttbaseref0:
; CHECK-NEXT:           .byte	1                       # Call site Encoding = uleb128
; CHECK-NEXT:           .uleb128 .Laction_table_base0-.Lcst_begin0
; CHECK-NEXT:         .Lcst_begin0:
; CHECK-NEXT:           .uleb128 .Ltmp0-.Lfunc_begin0   # >> Call Site 1 <<
; CHECK-NEXT:           .uleb128 .Ltmp1-.Ltmp0          #   Call between .Ltmp0 and .Ltmp1
; CHECK-NEXT:           .uleb128 .Ltmp2-main.__part.2   #     jumps to .Ltmp2
; CHECK-NEXT:           .byte	3                       #   On action: 2
; CHECK-NEXT:           .p2align	2
; CHECK-NEXT:         .Lexception1:

; CHECK-NON-PIC-NEXT:   .byte	0                       # @LPStart Encoding = absptr
; CHECK-NON-PIC-X64-NEXT: .quad	main.__part.2
; CHECK-NON-PIC-X32-NEXT: .long	main.__part.2

; CHECK-PIC-NEXT:       .byte	16                      # @LPStart Encoding = pcrel
; CHECK-PIC-NEXT:     [[DOT:\.Ltmp[0-9]+]]:
; CHECK-PIC-X64-NEXT:   .quad	main.__part.2-[[DOT]]
; CHECK-PIC-X32-NEXT:   .long	main.__part.2-[[DOT]]

; CHECK-NON-PIC-SMALL-NEXT:  .byte	3               # @TType Encoding = udata4
; CHECK-NON-PIC-MEDIUM-NEXT: .byte	0               # @TType Encoding = absptr
; CHECK-NON-PIC-LARGE-NEXT:  .byte	0               # @TType Encoding = absptr

; CHECK-PIC-SMALL-NEXT: .byte 155                       # @TType Encoding = indirect pcrel sdata4
; CHECK-PIC-MEDIUM-NEXT:.byte 155                       # @TType Encoding = indirect pcrel sdata4
; CHECK-PIC-LARGE-NEXT: .byte 156                       # @TType Encoding = indirect pcrel sdata8

; CHECK-NEXT:           .uleb128 .Lttbase0-.Lttbaseref1
; CHECK-NEXT:         .Lttbaseref1:
; CHECK-NEXT:           .byte	1                       # Call site Encoding = uleb128
; CHECK-NEXT:           .uleb128 .Laction_table_base0-.Lcst_begin1
; CHECK-NEXT:         .Lcst_begin1:
; CHECK-NEXT:           .p2align 2
; CHECK-NEXT:         .Lexception2:

; CHECK-NON-PIC-NEXT:   .byte	0                       # @LPStart Encoding = absptr
; CHECK-NON-PIC-X64-NEXT: .quad	main.__part.2
; CHECK-NON-PIC-X32-NEXT: .long	main.__part.2

; CHECK-PIC-NEXT:       .byte	16                      # @LPStart Encoding = pcrel
; CHECK-PIC-NEXT:     [[DOT:\.Ltmp[0-9]+]]:
; CHECK-PIC-X64-NEXT:   .quad	main.__part.2-[[DOT]]
; CHECK-PIC-X32-NEXT:   .long	main.__part.2-[[DOT]]

; CHECK-NON-PIC-SMALL-NEXT:  .byte	3               # @TType Encoding = udata4
; CHECK-NON-PIC-MEDIUM-NEXT: .byte	0               # @TType Encoding = absptr
; CHECK-NON-PIC-LARGE-NEXT:  .byte	0               # @TType Encoding = absptr

; CHECK-PIC-SMALL-NEXT:  .byte 155                      # @TType Encoding = indirect pcrel sdata4
; CHECK-PIC-MEDIUM-NEXT: .byte 155                      # @TType Encoding = indirect pcrel sdata4
; CHECK-PIC-LARGE-NEXT:  .byte 156                      # @TType Encoding = indirect pcrel sdata8

; CHECK-NEXT:           .uleb128 .Lttbase0-.Lttbaseref2
; CHECK-NEXT:         .Lttbaseref2:
; CHECK-NEXT:           .byte	1                       # Call site Encoding = uleb128
; CHECK-NEXT:           .uleb128 .Laction_table_base0-.Lcst_begin2
; CHECK-NEXT:         .Lcst_begin2:
; CHECK-NEXT:           .uleb128 main.__part.2-main.__part.2          # >> Call Site 2 <<
; CHECK-NEXT:           .uleb128 .LBB_END0_2-main.__part.2     #   Call between main.__part.2 and .LBB_END0_2
; CHECK-NEXT:           .byte	0                       #     has no landing pad
; CHECK-NEXT:           .byte	0                       #   On action: cleanup
; CHECK-NEXT:         .Laction_table_base0:
; CHECK-NEXT:           .byte	0                       # >> Action Record 1 <<
; CHECK-NEXT:                                           #   Cleanup
; CHECK-NEXT:           .byte	0                       #   No further actions
; CHECK-NEXT:           .byte	1                       # >> Action Record 2 <<
; CHECK-NEXT:                                           #   Catch TypeInfo 1
; CHECK-NEXT:           .byte	125                     #   Continue to action 1
; CHECK-NEXT:           .p2align 2
; CHECK-NEXT:                                           # >> Catch TypeInfos <<

; CHECK-NON-PIC-SMALL-NEXT:  .long _ZTIi                # TypeInfo 1
; CHECK-NON-PIC-MEDIUM-NEXT: .quad _ZTIi                # TypeInfo 1
; CHECK-NON-PIC-LARGE-NEXT:  .quad _ZTIi                # TypeInfo 1

; CHECK-PIC-NEXT:     [[DOT:\.Ltmp[0-9]+]]:
; CHECK-PIC-SMALL-NEXT:  .long .L_ZTIi.DW.stub-[[DOT]]
; CHECK-PIC-MEDIUM-NEXT: .long .L_ZTIi.DW.stub-[[DOT]]
; CHECK-PIC-LARGE-NEXT:  .quad .L_ZTIi.DW.stub-[[DOT]]

; CHECK-NEXT:         .Lttbase0:
; CHECK-NEXT:           .p2align 2
; CHECK-NEXT:                                           # -- End function
