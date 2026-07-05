; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s --check-prefix=NO-OPTION
; RUN: llc < %s -mtriple=x86_64-- -disable-tail-calls | FileCheck %s --check-prefix=DISABLE-TRUE
; RUN: llc < %s -mtriple=x86_64-- -disable-tail-calls=false | FileCheck %s --check-prefix=DISABLE-FALSE

; Check that command line option "-disable-tail-calls" overrides function
; attribute "disable-tail-calls".

; NO-OPTION-LABEL: {{\_?}}func_attr
; NO-OPTION: callq {{\_?}}callee

; DISABLE-FALSE-LABEL: {{\_?}}func_attr
; DISABLE-FALSE: jmp {{\_?}}callee

; DISABLE-TRUE-LABEL: {{\_?}}func_attr
; DISABLE-TRUE: callq {{\_?}}callee

define tailcc i32 @func_attr(i32 %a) #0 {
entry:
  %call = tail call tailcc i32 @callee(i32 %a)
  ret i32 %call
}

; NO-OPTION-LABEL: {{\_?}}func_noattr
; NO-OPTION: jmp {{\_?}}callee

; DISABLE-FALSE-LABEL: {{\_?}}func_noattr
; DISABLE-FALSE: jmp {{\_?}}callee

; DISABLE-TRUE-LABEL: {{\_?}}func_noattr
; DISABLE-TRUE: callq {{\_?}}callee

define tailcc i32 @func_noattr(i32 %a) {
entry:
  %call = tail call tailcc i32 @callee(i32 %a)
  ret i32 %call
}

declare tailcc i32 @callee(i32)

attributes #0 = { "disable-tail-calls"="true" }
