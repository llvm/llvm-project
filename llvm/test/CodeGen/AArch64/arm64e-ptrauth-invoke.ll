; RUN: llc -mtriple arm64e-apple-darwin -o - %s | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: llc -mtriple arm64e-apple-darwin -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK,FISEL

; CHECK-LABEL: _test_invoke_ia_0:
; CHECK-NEXT: [[FNBEGIN:L.*]]:
; CHECK-NEXT:  .cfi_startproc
; CHECK-NEXT:  .cfi_personality 155, ___gxx_personality_v0
; CHECK-NEXT:  .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; CHECK-NEXT: ; %bb.0:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x20, x19, [sp, #-32]!
; CHECK-NEXT:  stp x29, x30, [sp, #16]
; CHECK-NEXT:  .cfi_def_cfa_offset 32
; CHECK-NEXT:  .cfi_offset w30, -8
; CHECK-NEXT:  .cfi_offset w29, -16
; CHECK-NEXT:  .cfi_offset w19, -24
; CHECK-NEXT:  .cfi_offset w20, -32
; CHECK-NEXT: [[PRECALL:L.*]]:
; CHECK-NEXT:  blraaz x0
; CHECK-NEXT: [[POSTCALL:L.*]]:
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  mov x19, x0
; CHECK-NEXT:  bl _foo
; CHECK-NEXT:  mov x0, x19
; CHECK-NEXT: [[EXITBB:LBB[0-9_]+]]:
; CHECK-NEXT:  ldp x29, x30, [sp, #16]
; CHECK-NEXT:  ldp x20, x19, [sp], #32
; CHECK-NEXT:  retab
; CHECK-NEXT: [[LPADBB:LBB[0-9_]+]]:
; CHECK-NEXT: [[LPAD:L.*]]:
; CHECK-NEXT:  bl _foo
; CHECK-NEXT:  mov w0, #-1
; CHECK-NEXT:  b [[EXITBB]]

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK:       .uleb128 [[POSTCALL]]-[[PRECALL]] ;   Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:  .uleb128 [[LPAD]]-[[FNBEGIN]]     ;     jumps to [[LPAD]]
; CHECK-NEXT:  .byte 0                           ;   On action: cleanup

define i32 @test_invoke_ia_0(i32 ()* %arg0) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %tmp0 = invoke i32 %arg0() [ "ptrauth"(i32 0, i64 0) ] to label %continuebb
            unwind label %unwindbb

unwindbb:
  %tmp1 = landingpad { i8*, i32 } cleanup
  call void @foo()
  ret i32 -1

continuebb:
  call void @foo()
  ret i32 %tmp0
}

; CHECK-LABEL: _test_invoke_ia_0_direct:
; CHECK-NEXT: [[FNBEGIN:L.*]]:
; CHECK-NEXT:  .cfi_startproc
; CHECK-NEXT:  .cfi_personality 155, ___gxx_personality_v0
; CHECK-NEXT:  .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; CHECK-NEXT: ; %bb.0:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x20, x19, [sp, #-32]!
; CHECK-NEXT:  stp x29, x30, [sp, #16]
; CHECK-NEXT:  .cfi_def_cfa_offset 32
; CHECK-NEXT:  .cfi_offset w30, -8
; CHECK-NEXT:  .cfi_offset w29, -16
; CHECK-NEXT:  .cfi_offset w19, -24
; CHECK-NEXT:  .cfi_offset w20, -32
; CHECK-NEXT: [[PRECALL:L.*]]:
; CHECK-NEXT:  bl _baz
; CHECK-NEXT: [[POSTCALL:L.*]]:
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  mov x19, x0
; CHECK-NEXT:  bl _foo
; CHECK-NEXT:  mov x0, x19
; CHECK-NEXT: [[EXITBB:LBB[0-9_]+]]:
; CHECK-NEXT:  ldp x29, x30, [sp, #16]
; CHECK-NEXT:  ldp x20, x19, [sp], #32
; CHECK-NEXT:  retab
; CHECK-NEXT: [[LPADBB:LBB[0-9_]+]]:
; CHECK-NEXT: [[LPAD:L.*]]:
; CHECK-NEXT:  bl _foo
; CHECK-NEXT:  mov w0, #-1
; CHECK-NEXT:  b [[EXITBB]]

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK:       .uleb128 [[POSTCALL]]-[[PRECALL]] ;   Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:  .uleb128 [[LPAD]]-[[FNBEGIN]]     ;     jumps to [[LPAD]]
; CHECK-NEXT:  .byte 0                           ;   On action: cleanup

define i32 @test_invoke_ia_0_direct() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %tmp0 = invoke i32 bitcast ({ i8*, i32, i64, i64 }* @baz.ptrauth to i32 ()*)() [ "ptrauth"(i32 0, i64 0) ] to label %continuebb
            unwind label %unwindbb

unwindbb:
  %tmp1 = landingpad { i8*, i32 } cleanup
  call void @foo()
  ret i32 -1

continuebb:
  call void @foo()
  ret i32 %tmp0
}

@baz.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32()* @baz to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth"

@_ZTIPKc = external constant i8*
@hello_str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

; CHECK-LABEL: _test_invoke_ib_42_catch:
; CHECK-NEXT: [[FNBEGIN:L.*]]:
; CHECK-NEXT:         .cfi_startproc
; CHECK-NEXT:         .cfi_personality 155, ___gxx_personality_v0
; CHECK-NEXT:         .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; CHECK-NEXT: ; %bb.0:
; CHECK-NEXT:         pacibsp
; CHECK-NEXT:         stp x20, x19, [sp, #-32]!
; CHECK-NEXT:         stp x29, x30, [sp, #16]
; CHECK-NEXT:         .cfi_def_cfa_offset 32
; CHECK-NEXT:         .cfi_offset w30, -8
; CHECK-NEXT:         .cfi_offset w29, -16
; CHECK-NEXT:         .cfi_offset w19, -24
; CHECK-NEXT:         .cfi_offset w20, -32
; CHECK-NEXT:         mov x19, x0
; CHECK-NEXT:         mov w0, #8
; CHECK-NEXT:         bl ___cxa_allocate_exception
; CHECK-NEXT: Lloh{{.*}}:
; CHECK-NEXT:         adrp x8, l_hello_str@PAGE
; CHECK-NEXT: Lloh{{.*}}:
; CHECK-NEXT:         add x8, x8, l_hello_str@PAGEOFF
; CHECK-NEXT:         str x8, [x0]
; CHECK-NEXT: [[PRECALL:L.*]]:
; CHECK-NEXT: Lloh{{.*}}:
; CHECK-NEXT:         adrp x1, __ZTIPKc@GOTPAGE
; CHECK-NEXT: Lloh{{.*}}:
; CHECK-NEXT:         ldr x1, [x1, __ZTIPKc@GOTPAGEOFF]
; CHECK-NEXT:         mov w8, #42
; CHECK-NEXT:         mov x2, #0
; CHECK-NEXT:         blrab x19, x8
; CHECK-NEXT: [[POSTCALL:L.*]]:
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:         brk #0x1
; CHECK-NEXT: [[LPADBB:LBB[0-9_]+]]:
; CHECK-NEXT: [[LPAD:L.*]]:
; CHECK-NEXT:         mov x19, x1

; SDAG-NEXT:          bl ___cxa_begin_catch
; SDAG-NEXT:          cmp     w19, #2

; FISEL-NEXT:         mov w20, #2
; FISEL-NEXT:         bl ___cxa_begin_catch
; FISEL-NEXT:         cmp w19, w20

; CHECK-NEXT:         b.ne [[EXITBB:LBB[0-9_]+]]
; CHECK-NEXT: ; %bb.3:
; CHECK-NEXT:         bl _bar
; CHECK-NEXT: [[EXITBB]]:
; CHECK-NEXT:         bl _foo
; CHECK-NEXT:         bl ___cxa_end_catch
; CHECK-NEXT:         ldp x29, x30, [sp, #16]
; CHECK-NEXT:         ldp x20, x19, [sp], #32
; CHECK-NEXT:         retab
; CHECK-NEXT:         .loh {{.*}}
; CHECK-NEXT:         .loh {{.*}}
; CHECK-NEXT: [[FNEND:L.*]]:

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK-NEXT:         .byte   255                     ; @LPStart Encoding = omit
; CHECK-NEXT:         .byte   155                     ; @TType Encoding = indirect pcrel sdata4
; CHECK-NEXT:         .uleb128 [[TT:L.*]]-[[TTREF:L.*]]
; CHECK-NEXT: [[TTREF]]:
; CHECK-NEXT:         .byte   1                       ; Call site Encoding = uleb128
; CHECK-NEXT:         .uleb128 [[CSEND:L.*]]-[[CSBEGIN:L.*]]
; CHECK-NEXT: [[CSBEGIN]]:
; CHECK-NEXT:         .uleb128 [[FNBEGIN]]-[[FNBEGIN]]  ; >> Call Site 1 <<
; CHECK-NEXT:         .uleb128 [[PRECALL]]-[[FNBEGIN]]  ;   Call between [[FNBEGIN]] and [[PRECALL]]
; CHECK-NEXT:         .byte   0                         ;     has no landing pad
; CHECK-NEXT:         .byte   0                         ;   On action: cleanup
; CHECK-NEXT:         .uleb128 [[PRECALL]]-[[FNBEGIN]]  ; >> Call Site 2 <<
; CHECK-NEXT:         .uleb128 [[POSTCALL]]-[[PRECALL]] ;   Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:         .uleb128 [[LPAD]]-[[FNBEGIN]]     ;     jumps to [[LPAD]]
; CHECK-NEXT:         .byte   3                         ;   On action: 2
; CHECK-NEXT:         .uleb128 [[POSTCALL]]-[[FNBEGIN]] ; >> Call Site 3 <<
; CHECK-NEXT:         .uleb128 [[FNEND]]-[[POSTCALL]]   ;   Call between [[POSTCALL]] and [[FNEND]]
; CHECK-NEXT:         .byte   0                         ;     has no landing pad
; CHECK-NEXT:         .byte   0                         ;   On action: cleanup
; CHECK-NEXT: [[CSEND]]:

; CHECK-NEXT:          .byte   1                       ; >> Action Record 1 <<
; CHECK-NEXT:                                          ;   Catch TypeInfo 1
; CHECK-NEXT:          .byte   0                       ;   No further actions
; CHECK-NEXT:          .byte   2                       ; >> Action Record 2 <<
; CHECK-NEXT:                                          ;   Catch TypeInfo 2
; CHECK-NEXT:          .byte   125                     ;   Continue to action 1
; CHECK-NEXT:          .p2align   2
; CHECK-NEXT:                                          ; >> Catch TypeInfos <<
; CHECK-NEXT: [[TI:L.*]]:                              ; TypeInfo 2
; CHECK-NEXT:          .long   __ZTIPKc@GOT-[[TI]]
; CHECK-NEXT:          .long   0                       ; TypeInfo 1

; CHECK-NEXT: [[TT]]:

define void @test_invoke_ib_42_catch(void(i8*, i8*, i8*)* %fptr) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %tmp0 = call i8* @__cxa_allocate_exception(i64 8)
  %tmp1 = bitcast i8* %tmp0 to i8**
  store i8* getelementptr inbounds ([6 x i8], [6 x i8]* @hello_str, i64 0, i64 0), i8** %tmp1, align 8
  invoke void %fptr(i8* %tmp0, i8* bitcast (i8** @_ZTIPKc to i8*), i8* null) [ "ptrauth"(i32 1, i64 42) ]
          to label %continuebb unwind label %catchbb

catchbb:
  %tmp2 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIPKc to i8*)
          catch i8* null
  %tmp3 = extractvalue { i8*, i32 } %tmp2, 0
  %tmp4 = extractvalue { i8*, i32 } %tmp2, 1
  %tmp5 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIPKc to i8*))
  %tmp6 = icmp eq i32 %tmp4, %tmp5
  %tmp7 = call i8* @__cxa_begin_catch(i8* %tmp3)
  br i1 %tmp6, label %PKc_catchbb, label %any_catchbb

PKc_catchbb:
  call void @bar(i8* %tmp7)
  br label %any_catchbb

any_catchbb:
  call void @foo()
  call void @__cxa_end_catch()
  ret void

continuebb:
  unreachable
}

declare void @foo()
declare void @bar(i8*)
declare i32 @baz()

declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_allocate_exception(i64)
declare void @__cxa_throw(i8*, i8*, i8*)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()

attributes #0 = { nounwind "ptrauth-returns" }
