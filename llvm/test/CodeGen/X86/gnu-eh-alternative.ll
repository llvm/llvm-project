; RUN: llc -verify-machineinstrs -mtriple x86_64-pc-linux-gnu -filetype=asm < %s | \
; RUN:   FileCheck --check-prefixes=ASM,ULEB128 %s
; RUN: llc -verify-machineinstrs -mtriple x86_64-pc-linux-gnu -use-leb128-directives=true -filetype=asm < %s | \
; RUN:   FileCheck --check-prefixes=ASM,ULEB128 %s
; RUN: llc -verify-machineinstrs -mtriple x86_64-pc-linux-gnu -use-leb128-directives=false -filetype=asm < %s | \
; RUN:   FileCheck --check-prefixes=ASM,NO128 %s

@_ZTIi = external dso_local constant i8*

define dso_local i32 @main() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca i8*, align 8
  %ehselector.slot = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %exception = call i8* @__cxa_allocate_exception(i64 4) #1
  %0 = bitcast i8* %exception to i32*
  store i32 1, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #2
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot, align 8
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot, align 4
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot, align 8
  %4 = call i8* @__cxa_begin_catch(i8* %exn) #1
  store i32 2, i32* %retval, align 4
  call void @__cxa_end_catch()
  br label %return

try.cont:                                         ; No predecessors!
  store i32 1, i32* %retval, align 4
  br label %return

return:                                           ; preds = %try.cont, %catch
  %5 = load i32, i32* %retval, align 4
  ret i32 %5

unreachable:                                      ; preds = %entry
  unreachable
}

; ASM:     GCC_except_table0:
; ASM:     .Lexception0:
; ASM:     	.byte	255                             # @LPStart Encoding = omit
; ASM:     	.byte	3                               # @TType Encoding = udata4

; NO128:   	.byte	49
; NO128:   	.byte	3                               # Call site Encoding = udata4
; NO128:   	.byte	39
; NO128:    .long	.Lfunc_begin0-.Lfunc_begin0     # >> Call Site 1 <<
; NO128:    .long	.Ltmp0-.Lfunc_begin0            #   Call between .Lfunc_begin0 and .Ltmp0
; NO128:    .long	0                               #     has no landing pad
; NO128:    .byte	0                               #   On action: cleanup
; NO128:    .long	.Ltmp0-.Lfunc_begin0            # >> Call Site 2 <<
; NO128:    .long	.Ltmp1-.Ltmp0                   #   Call between .Ltmp0 and .Ltmp1
; NO128:    .long	.Ltmp2-.Lfunc_begin0            #     jumps to .Ltmp2
; NO128:    .byte	1                               #   On action: 1
; NO128:    .long	.Ltmp1-.Lfunc_begin0            # >> Call Site 3 <<
; NO128:    .long	.Lfunc_end0-.Ltmp1              #   Call between .Ltmp1 and .Lfunc_end0
; NO128:    .long	0                               #     has no landing pad
; NO128:    .byte	0                               #   On action: cleanup

; ULEB128: 	.uleb128 .Lttbase0-.Lttbaseref0
; ULEB128: .Lttbaseref0:
; ULEB128: 	.byte	1                               # Call site Encoding = uleb128
; ULEB128: 	.uleb128 .Lcst_end0-.Lcst_begin0
; ULEB128: .Lcst_begin0:
; ULEB128: 	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
; ULEB128: 	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
; ULEB128: 	.byte	0                               #     has no landing pad
; ULEB128: 	.byte	0                               #   On action: cleanup
; ULEB128: 	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
; ULEB128: 	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
; ULEB128: 	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
; ULEB128: 	.byte	1                               #   On action: 1
; ULEB128: 	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
; ULEB128: 	.uleb128 .Lfunc_end0-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end0
; ULEB128: 	.byte	0                               #     has no landing pad
; ULEB128: 	.byte	0                               #   On action: cleanup

; ASM:     .Lcst_end0:
; ASM:     	.byte	1                               # >> Action Record 1 <<
; ASM:                                             #   Catch TypeInfo 1
; ASM:     	.byte	0                               #   No further actions
; ASM:     	.p2align	2
; ASM:                                             # >> Catch TypeInfos <<
; ASM:     	.long	0                               # TypeInfo 1
; ASM:     .Lttbase0:
; ASM:     	.p2align	2

declare dso_local i8* @__cxa_allocate_exception(i64)
declare dso_local void @__cxa_throw(i8*, i8*, i8*)
declare dso_local i32 @__gxx_personality_v0(...)
declare dso_local i8* @__cxa_begin_catch(i8*)
declare dso_local void @__cxa_end_catch()
