; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=i386-unknown-unknown < %s | FileCheck %s   --check-prefix=X86

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; foo
;; -----
;; Checks ENDBR insertion after return twice functions.
;; setjmp, sigsetjmp, savectx, vfork, getcontext
;; setzx is not return twice function, should not followed by endbr.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; X64:           callq   setjmp
; X64-NEXT:      endbr64
; X64:           callq   setzx
; X64-NEXT:      xorl
; X64:           callq   sigsetjmp
; X64-NEXT:      endbr64
; X64:           callq   savectx
; X64-NEXT:      endbr64
; X64:           callq   vfork
; X64-NEXT:      endbr64
; X64:           callq   getcontext
; X64-NEXT:      endbr64

; X86:           calll   setjmp
; X86-NEXT:      endbr32
; X86:           calll   setzx
; X86-NEXT:      xorl
; X86:           calll   sigsetjmp
; X86-NEXT:      endbr32
; X86:           calll   savectx
; X86-NEXT:      endbr32
; X86:           calll   vfork
; X86-NEXT:      endbr32
; X86:           calll   getcontext
; X86-NEXT:      endbr32

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 {
entry:
  %call = call i32 (i32, ...) bitcast (i32 (...)* @setjmp to i32 (i32, ...)*)(i32 0) #1
  %call1 = call i32 (i32, ...) bitcast (i32 (...)* @setzx to i32 (i32, ...)*)(i32 0)
  %call2 = call i32 (i32, ...) bitcast (i32 (...)* @sigsetjmp to i32 (i32, ...)*)(i32 0) #1
  %call3 = call i32 (i32, ...) bitcast (i32 (...)* @setzx to i32 (i32, ...)*)(i32 0)
  %call4 = call i32 (i32, ...) bitcast (i32 (...)* @savectx to i32 (i32, ...)*)(i32 0) #1
  %call5 = call i32 @vfork() #1
  %call6 = call i32 (i32, ...) bitcast (i32 (...)* @setzx to i32 (i32, ...)*)(i32 0)
  %call7 = call i32 (i32, ...) bitcast (i32 (...)* @getcontext to i32 (i32, ...)*)(i32 0) #1
  ret void
}

; Function Attrs: returns_twice
declare dso_local i32 @setjmp(...) #1

declare dso_local i32 @setzx(...)

; Function Attrs: returns_twice
declare dso_local i32 @sigsetjmp(...) #1

; Function Attrs: returns_twice
declare dso_local i32 @savectx(...) #1

; Function Attrs: returns_twice
declare dso_local i32 @vfork() #1

; Function Attrs: returns_twice
declare dso_local i32 @getcontext(...) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { returns_twice }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"cf-protection-return", i32 1}
!2 = !{i32 8, !"cf-protection-branch", i32 1}
