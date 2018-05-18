; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=ALL --check-prefix=X86_64
; RUN: llc -mtriple=i386-unknown-unknown < %s | FileCheck %s --check-prefix=ALL --check-prefix=X86
; RUN: llc -mtriple i386-windows-gnu -exception-model sjlj < %s | FileCheck %s --check-prefix=SJLJ

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test1
;; -----
;; Checks ENDBR insertion in case of switch case statement.
;; Also since the function is not internal, make sure that endbr32/64 was 
;; added at the beginning of the function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i8 @test1(){
; ALL-LABEL:   test1
; X86_64:      endbr64
; X86:         endbr32
; ALL:         jmp{{q|l}} *
; ALL:         .LBB0_1:
; X86_64-NEXT: endbr64
; X86-NEXT:    endbr32
; ALL:         .LBB0_2:
; X86_64-NEXT: endbr64
; X86-NEXT:    endbr32
entry:
  %0 = select i1 undef, i8* blockaddress(@test1, %bb), i8* blockaddress(@test1, %bb6) ; <i8*> [#uses=1]
  indirectbr i8* %0, [label %bb, label %bb6]

bb:                                               ; preds = %entry
  ret i8 1

bb6:                                              ; preds = %entry
  ret i8 2
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test2
;; -----
;; Checks NOTRACK insertion in case of switch case statement.
;; Check that there is no ENDBR insertion in the following case statements.
;; Also since the function is not internal, ENDBR instruction should be
;; added to its first basic block.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i32 @test2(i32 %a) {
; ALL-LABEL:   test2
; X86_64:      endbr64
; X86:         endbr32
; ALL:         notrack jmp{{q|l}} *
; X86_64-NOT:      endbr64
; X86-NOT:         endbr32
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry
  store i32 5, i32* %retval, align 4
  br label %return

sw.bb1:                                           ; preds = %entry
  store i32 7, i32* %retval, align 4
  br label %return

sw.bb2:                                           ; preds = %entry
  store i32 2, i32* %retval, align 4
  br label %return

sw.bb3:                                           ; preds = %entry
  store i32 32, i32* %retval, align 4
  br label %return

sw.bb4:                                           ; preds = %entry
  store i32 73, i32* %retval, align 4
  br label %return

sw.default:                                       ; preds = %entry
  store i32 0, i32* %retval, align 4
  br label %return

return:                                           ; preds = %sw.default, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test3
;; -----
;; Checks ENDBR insertion in case of indirect call instruction.
;; The new instruction should be added to the called function (test6)
;; although it is internal.
;; Also since the function is not internal, ENDBR instruction should be
;; added to its first basic block.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define void @test3() {
; ALL-LABEL:   test3
; X86_64:      endbr64
; X86:         endbr32
; ALL:         call{{q|l}} *
entry:
  %f = alloca i32 (...)*, align 8
  store i32 (...)* bitcast (i32 (i32)* @test6 to i32 (...)*), i32 (...)** %f, align 8
  %0 = load i32 (...)*, i32 (...)** %f, align 8
  %call = call i32 (...) %0()
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test4
;; -----
;; Checks ENDBR insertion in case of setjmp-like function calls.
;; Also since the function is not internal, ENDBR instruction should be
;; added to its first basic block.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

@buf = internal global [5 x i8*] zeroinitializer
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.stacksave()
declare i32 @llvm.eh.sjlj.setjmp(i8*)

define i32 @test4() {
; ALL-LABEL:   test4
; X86_64:      endbr64
; X86:         endbr32
; ALL:         .LBB3_3:
; X86_64-NEXT: endbr64
; X86-NEXT:    endbr32
  %fp = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %fp, i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @buf, i64 0, i64 0), align 16
  %sp = tail call i8* @llvm.stacksave()
  store i8* %sp, i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @buf, i64 0, i64 2), align 16
  %r = tail call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([5 x i8*]* @buf to i8*))
  ret i32 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test5
;; -----
;; Checks ENDBR insertion in case of internal function.
;; Since the function is internal and its address was not taken,
;; make sure that endbr32/64 was not added at the beginning of the 
;; function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define internal i8 @test5(){
; ALL-LABEL:   test5
; X86_64-NOT:      endbr64
; X86-NOT:         endbr32
  ret i8 1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test6
;; -----
;; Checks ENDBR insertion in case of function that its was address taken.
;; Since the function's address was taken by test3() and despite being
;; internal, check for added endbr32/64 at the beginning of the function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define internal i32 @test6(i32 %a) {
; ALL-LABEL:   test6
; X86_64:      endbr64
; X86:         endbr32
  ret i32 1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test7
;; -----
;; Checks ENDBR insertion in case of non-intrenal function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i32 @test7() {
; ALL-LABEL:   test7
; X86_64:      endbr64
; X86:         endbr32
  ret i32 1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test8
;; -----
;; Checks that NO TRACK prefix is not added for indirect jumps to a jump-
;; table that was created for SJLJ dispatch.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

declare void @_Z20function_that_throwsv()
declare i32 @__gxx_personality_sj0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()

define void @test8() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
;SJLJ-LABEL:    test8
;SJLJ-NOT:      ds
entry:
  invoke void @_Z20function_that_throwsv()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"cf-protection-branch", i32 1}
