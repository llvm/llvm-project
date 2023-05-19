; RUN: llc %s -o - | FileCheck %s
; RUN: llc < %s -stop-after=prologepilog | FileCheck %s --check-prefix=PEI

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__clang_call_terminate = comdat any

@_ZL11ShouldThrow = internal unnamed_addr global i1 false, align 1
@_ZTIi = external constant i8*
@str = private unnamed_addr constant [20 x i8] c"Threw an exception!\00"

; Function Attrs: uwtable
define void @_Z6throwsv() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {

; CHECK-LABEL:   _Z6throwsv:
; CHECK:         popq %rbp
; CHECK-NEXT:    .cfi_def_cfa %rsp, 8
; CHECK-NEXT:    retq
; CHECK-NEXT:    .LBB0_1:
; CHECK-NEXT:    .cfi_def_cfa %rbp, 16

; PEI-LABEL: name: _Z6throwsv
; PEI:         frame-setup PUSH64r killed $rbp, implicit-def $rsp, implicit $rsp
; PEI-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 16
; PEI-NEXT:    frame-setup CFI_INSTRUCTION offset $rbp, -16
; PEI-NEXT:    $rbp = frame-setup MOV64rr $rsp
; PEI-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_register $rbp
; PEI-NEXT:    frame-setup PUSH64r killed $rbx, implicit-def $rsp, implicit $rsp
; PEI-NEXT:    frame-setup PUSH64r undef $rax, implicit-def $rsp, implicit $rsp
; PEI-NEXT:    {{^ +}}CFI_INSTRUCTION offset $rbx, -24
; PEI:         bb.4.try.cont:
; PEI-NEXT:      $rsp = frame-destroy ADD64ri32 $rsp, 8, implicit-def dead $eflags
; PEI-NEXT:      $rbx = frame-destroy POP64r implicit-def $rsp, implicit $rsp
; PEI-NEXT:      $rbp = frame-destroy POP64r implicit-def $rsp, implicit $rsp
; PEI-NEXT:      frame-destroy CFI_INSTRUCTION def_cfa $rsp, 8
; PEI-NEXT:      RET 0

entry:
  %.b5 = load i1, i1* @_ZL11ShouldThrow, align 1
  br i1 %.b5, label %if.then, label %try.cont

if.then:                                          ; preds = %entry
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  %0 = bitcast i8* %exception to i32*
  store i32 1, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %if.then
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2)
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @str, i64 0, i64 0))
  invoke void @__cxa_rethrow()
          to label %unreachable unwind label %lpad1

lpad1:                                            ; preds = %lpad
  %4 = landingpad { i8*, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

try.cont:                                         ; preds = %entry
  ret void

eh.resume:                                        ; preds = %lpad1
  resume { i8*, i32 } %4

terminate.lpad:                                   ; preds = %lpad1
  %5 = landingpad { i8*, i32 }
          catch i8* null
  %6 = extractvalue { i8*, i32 } %5, 0
  tail call void @__clang_call_terminate(i8* %6)
  unreachable

unreachable:                                      ; preds = %lpad, %if.then
  unreachable
}

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

; Function Attrs: noinline noreturn nounwind
declare void @__clang_call_terminate(i8*)

declare void @_ZSt9terminatev()

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly)

attributes #0 = { "frame-pointer"="all" }

