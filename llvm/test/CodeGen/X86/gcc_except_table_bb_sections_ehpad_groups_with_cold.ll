; Check that when all exception handling blocks are cold, they get grouped with the cold bbs.
; RUN: echo '!main' > %t
; RUN: echo '!!0' >> %t
; RUN: llc -function-sections -basic-block-sections=%t -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s
@_ZTIi = external constant i8*

define i32 @main() uwtable optsize ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; Verify that each basic block section gets its own LSDA exception symbol.
;
; CHECK-LABEL:  main:
; CHECK-NEXT:    .Lfunc_begin0:
; CHECK-NEXT:    .cfi_startproc
; PersonalityEncoding = dwarf::DW_EH_PE_udata4
; CHECK-NEXT:    .cfi_personality 3, __gxx_personality_v0
; LSDAEncoding = dwarf::DW_EH_PE_udata4
; CHECK-NEXT:    .cfi_lsda 3, .Lexception0
; CHECK-LABEL:  .Ltmp0:
; CHECK-LABEL:  .Ltmp1:

; CHECK-NOT: .cfi_lsda

; CHECK-LABEL:  main.cold:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    .cfi_personality 3, __gxx_personality_v0
; CHECK-NEXT:    .cfi_lsda 3, .Lexception1
; CHECK-LABEL:  .Ltmp2:
; CHECK-LABEL:  .LBB_END0_2:

; CHECK-NOT: .cfi_lsda

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

; Verify that the exception table gets split across the two basic block sections.
;
; CHECK:       .section .gcc_except_table
; CHECK-NEXT:  .p2align 2
; CHECK-NEXT:  GCC_except_table0:
; CHECK-NEXT:  .Lexception0:
; CHECK-NEXT:    .byte	0                       # @LPStart Encoding = absptr
; CHECK-NEXT:    .quad	main.cold
; CHECK-NEXT:    .byte	3                       # @TType Encoding = udata4
; CHECK-NEXT:    .uleb128 .Lttbase0-.Lttbaseref0
; CHECK-NEXT:  .Lttbaseref0:
; CHECK-NEXT:    .byte	1                       # Call site Encoding = uleb128
; CHECK-NEXT:    .uleb128 .Laction_table_base0-.Lcst_begin0
; CHECK-NEXT:  .Lcst_begin0:
; CHECK-NEXT:    .uleb128 .Ltmp0-.Lfunc_begin0  # >> Call Site 1 <<
; CHECK-NEXT:    .uleb128 .Ltmp1-.Ltmp0         #   Call between .Ltmp0 and .Ltmp1
; CHECK-NEXT:    .uleb128 .Ltmp2-main.cold      #     jumps to .Ltmp2
; CHECK-NEXT:    .byte	3                       #   On action: 2
; CHECK-NEXT:    .p2align	2
; CHECK-NEXT:  .Lexception1:
; CHECK-NEXT:    .byte	0                       # @LPStart Encoding = absptr
; CHECK-NEXT:    .quad	main.cold
; CHECK-NEXT:    .byte	3                       # @TType Encoding = udata4
; CHECK-NEXT:    .uleb128 .Lttbase0-.Lttbaseref1
; CHECK-NEXT:  .Lttbaseref1:
; CHECK-NEXT:    .byte	1                       # Call site Encoding = uleb128
; CHECK-NEXT:    .uleb128 .Laction_table_base0-.Lcst_begin1
; CHECK-NEXT:  .Lcst_begin1:
; CHECK-NEXT:    .uleb128 main.cold-main.cold   # >> Call Site 2 <<
; CHECK-NEXT:    .uleb128 .LBB_END0_2-main.cold #   Call between main.cold and .LBB_END0_2
; CHECK-NEXT:    .byte	0                       #     has no landing pad
; CHECK-NEXT:    .byte	0                       #   On action: cleanup
; CHECK-NEXT:  .Laction_table_base0:
; CHECK-NEXT:    .byte	0                       # >> Action Record 1 <<
; CHECK-NEXT:                                   #   Cleanup
; CHECK-NEXT:    .byte	0                       #   No further actions
; CHECK-NEXT:    .byte	1                       # >> Action Record 2 <<
; CHECK-NEXT:                                   #   Catch TypeInfo 1
; CHECK-NEXT:    .byte	125                     #   Continue to action 1
; CHECK-NEXT:    .p2align	2
; CHECK-NEXT:                                   # >> Catch TypeInfos <<
; CHECK-NEXT:    .long	_ZTIi                   # TypeInfo 1
; CHECK-NEXT:  .Lttbase0:
; CHECK-NEXT:    .p2align	2
; CHECK-NEXT:                                   # -- End function
