; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=X86_64
; RUN: llc -mtriple=i386-unknown-unknown < %s | FileCheck %s --check-prefix=X86
; RUN: llc -mtriple i386-windows-gnu -exception-model sjlj -verify-machineinstrs=0 < %s | FileCheck %s --check-prefix=SJLJ
; RUN: llc -mtriple i386-windows-gnu -exception-model sjlj -verify-machineinstrs=0 < %s | FileCheck %s --check-prefix=NUM

; X86_64:       test_eh:                                # @test_eh
; X86_64-NEXT:  .Lfunc_begin0:
; X86_64:       # %bb.0:                                # %entry
; X86_64-NEXT:          endbr64
; X86_64-NEXT:          pushq   %rax
; X86_64:       .Ltmp0:
; X86_64-NEXT:          callq   _Z20function_that_throwsv
; X86_64-NEXT:  .Ltmp1:
; X86_64-NEXT:  # %bb.1:                                # %try.cont
; X86_64:               retq
; X86_64-NEXT:  .LBB0_2:                                # %lpad
; X86_64-NEXT:          .cfi_def_cfa_offset 16
; X86_64-NEXT:  .Ltmp2:
; X86_64-NEXT:          endbr64
; X86_64:               callq   __cxa_begin_catch


; X86:       test_eh:                                # @test_eh
; X86-NEXT:  .Lfunc_begin0:
; X86:       # %bb.0:                                # %entry
; X86-NEXT:          endbr32
; X86-NEXT:  .Ltmp0:
; X86:               calll   _Z20function_that_throwsv
; X86-NEXT:  .Ltmp1:
; X86-NEXT:  # %bb.1:                                # %try.cont
; X86-NEXT:          retl
; X86-NEXT:  .LBB0_2:                                # %lpad
; X86-NEXT:  .Ltmp2:
; X86-NEXT:          endbr32
; X86:               calll   __cxa_begin_catch

; NUM-COUNT-3: endbr32

; SJLJ:       test_eh:
; SJLJ-NEXT:  Lfunc_begin0:
; SJLJ-NEXT:  # %bb.0:                                # %entry
; SJLJ-NEXT:          endbr32
; SJLJ:               calll   __Unwind_SjLj_Register
; SJLJ:       Ltmp0:
; SJLJ:               calll   __Z20function_that_throwsv
; SJLJ:       LBB0_2:                                 # %try.cont
; SJLJ:               calll   __Unwind_SjLj_Unregister
; SJLJ:               retl

; SJLJ:       LBB0_3:
; SJLJ-NEXT:          endbr32
; SJLJ-NEXT:          leal
; SJLJ-NEXT:          movl
; SJLJ-NEXT:          cmpl
; SJLJ-NEXT:          jb      LBB0_4

; SJLJ:       LBB0_4:
; SJLJ-NEXT:          jmpl    *LJTI0_0(,%eax,4)

; SJLJ:       LBB0_6:                                 # %lpad
; SJLJ-NEXT:  Ltmp2:
; SJLJ-NEXT:          endbr32
; SJLJ:               calll   ___cxa_begin_catch
; SJLJ:               jmp     LBB0_2
; SJLJ:       LJTI0_0:
; SJLJ-NEXT:          .long   LBB0_6



declare void @_Z20function_that_throwsv()
declare i32 @__gxx_personality_sj0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define void @test_eh() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @_Z20function_that_throwsv()
          to label %try.cont unwind label %lpad

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

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"cf-protection-branch", i32 1}
