;; Test if we are still able to compile even when the personality routine is just an alias.

; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefixes=SYM,SYM32
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefixes=SYM,SYM64

@__xlcxx_personality_v1 = alias i32 (), ptr @__gxx_personality_v0
define i32 @__gxx_personality_v0() {
entry:
  ret i32 1
}

define dso_local signext i32 @_Z3foov() #0 personality ptr @__xlcxx_personality_v1 {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca ptr, align 8
  %ehselector.slot = alloca i32, align 4
  invoke void @_Z3barv()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  store ptr %1, ptr %exn.slot, align 8
  %2 = extractvalue { ptr, i32 } %0, 1
  store i32 %2, ptr %ehselector.slot, align 4
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load ptr, ptr %exn.slot, align 8
  br label %return

try.cont:                                         ; preds = %invoke.cont
  store i32 2, ptr %retval, align 4
  br label %return

return:                                           ; preds = %try.cont, %catch
  ret i32 1
}

declare void @_Z3barv()

;   SYM:    .globl	__gxx_personality_v0[DS]        # -- Begin function __gxx_personality_v0
;   SYM:  	.globl	.__gxx_personality_v0
;   SYM:  	.align	4
;   SYM:  	.csect __gxx_personality_v0[DS]
;   SYM:  __xlcxx_personality_v1:                 # @__gxx_personality_v0
; SYM32:  	.vbyte	4, .__gxx_personality_v0
; SYM32:  	.vbyte	4, TOC[TC0]
; SYM32:  	.vbyte	4, 0
; SYM64:  	.vbyte	8, .__gxx_personality_v0
; SYM64:  	.vbyte	8, TOC[TC0]
; SYM64:  	.vbyte	8, 0
;   SYM:  	.csect [PR],5
;   SYM:  .__gxx_personality_v0:
;   SYM:  .__xlcxx_personality_v1:
;   SYM:  # %bb.0:                                # %entry
;   SYM:  	li 3, 1
;   SYM:  	blr

;   SYM:    .csect .eh_info_table[RW],2
;   SYM:  __ehinfo.1:
;   SYM:  	.vbyte	4, 0
; SYM32:  	.align	2
; SYM32:  	.vbyte	4, GCC_except_table1
; SYM32:  	.vbyte	4, __xlcxx_personality_v1
; SYM64:  	.align	3
; SYM64:  	.vbyte	8, GCC_except_table1
; SYM64:  	.vbyte	8, __xlcxx_personality_v1
