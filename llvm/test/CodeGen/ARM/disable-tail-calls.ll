; RUN: llc < %s -mtriple=arm-unknown-unknown | FileCheck %s --check-prefix=NO-OPTION
; RUN: llc < %s -mtriple=arm-unknown-unknown -disable-tail-calls | FileCheck %s --check-prefix=DISABLE-TRUE
; RUN: llc < %s -mtriple=arm-unknown-unknown -disable-tail-calls=false | FileCheck %s --check-prefix=DISABLE-FALSE

; Check that command line option "-disable-tail-calls" overrides function
; attribute "disable-tail-calls".

; NO-OPTION-LABEL: {{\_?}}func_attr
; NO-OPTION: bl {{\_?}}callee

; DISABLE-FALSE-LABEL: {{\_?}}func_attr
; DISABLE-FALSE: b {{\_?}}callee

; DISABLE-TRUE-LABEL: {{\_?}}func_attr
; DISABLE-TRUE: bl {{\_?}}callee

define i32 @func_attr(i32 %a) #0 {
entry:
  %call = tail call i32 @callee(i32 %a)
  ret i32 %call
}

; NO-OPTION-LABEL: {{\_?}}func_noattr
; NO-OPTION: b {{\_?}}callee

; DISABLE-FALSE-LABEL: {{\_?}}func_noattr
; DISABLE-FALSE: b {{\_?}}callee

; DISABLE-TRUE-LABEL: {{\_?}}func_noattr
; DISABLE-TRUE: bl {{\_?}}callee

define i32 @func_noattr(i32 %a) {
entry:
  %call = tail call i32 @callee(i32 %a)
  ret i32 %call
}

declare i32 @callee(i32)

attributes #0 = { "disable-tail-calls"="true" }
