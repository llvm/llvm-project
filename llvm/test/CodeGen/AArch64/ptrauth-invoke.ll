; RUN: llc -mtriple arm64e-apple-darwin   -o - %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DARWIN,DARWIN-SDAG

; RUN: llc -mtriple aarch64 -mattr=+pauth -o - %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ELF,ELF-SDAG

; RUN: llc -mtriple arm64e-apple-darwin   -o - %s \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:  | FileCheck %s --check-prefixes=CHECK,DARWIN,DARWIN-GISEL

; RUN: llc -mtriple aarch64 -mattr=+pauth -o - %s \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:  | FileCheck %s --check-prefixes=CHECK,ELF,ELF-GISEL

; DARWIN-LABEL: _test_invoke_ia_0:
; DARWIN-NEXT: [[FNBEGIN:L.*]]:
; DARWIN-NEXT:  .cfi_startproc
; DARWIN-NEXT:  .cfi_personality 155, ___gxx_personality_v0
; DARWIN-NEXT:  .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; DARWIN-NEXT: ; %bb.0:
; DARWIN-NEXT:  stp x20, x19, [sp, #-32]!
; DARWIN-NEXT:  stp x29, x30, [sp, #16]
; DARWIN-NEXT:  .cfi_def_cfa_offset 32
; DARWIN-NEXT:  .cfi_offset w30, -8
; DARWIN-NEXT:  .cfi_offset w29, -16
; DARWIN-NEXT:  .cfi_offset w19, -24
; DARWIN-NEXT:  .cfi_offset w20, -32
; DARWIN-NEXT: [[PRECALL:L.*]]:
; DARWIN-NEXT:  blraaz x0

; DARWIN-SDAG-NEXT: [[POSTCALL:L.*]]:
; DARWIN-SDAG-NEXT: ; %bb.1:
; DARWIN-SDAG-NEXT:  mov x19, x0

; DARWIN-GISEL-NEXT:  mov x19, x0
; DARWIN-GISEL-NEXT: [[POSTCALL:L.*]]:

; DARWIN-NEXT: [[CALLBB:L.*]]:
; DARWIN-NEXT:  bl _foo
; DARWIN-NEXT:  mov x0, x19
; DARWIN-NEXT:  ldp x29, x30, [sp, #16]
; DARWIN-NEXT:  ldp x20, x19, [sp], #32
; DARWIN-NEXT:  ret
; DARWIN-NEXT: [[LPADBB:LBB[0-9_]+]]:
; DARWIN-NEXT: [[LPAD:L.*]]:
; DARWIN-NEXT:  mov w19, #-1
; DARWIN-NEXT:  b [[CALLBB]]

; ELF-LABEL: test_invoke_ia_0:
; ELF-NEXT: [[FNBEGIN:.L.*]]:
; ELF-NEXT:  .cfi_startproc
; ELF-NEXT:  .cfi_personality 156, DW.ref.__gxx_personality_v0
; ELF-NEXT:  .cfi_lsda 28, [[EXCEPT:.Lexception[0-9]+]]
; ELF-NEXT: // %bb.0:
; ELF-NEXT:  stp x30, x19, [sp, #-16]!
; ELF-NEXT:  .cfi_def_cfa_offset 16
; ELF-NEXT:  .cfi_offset w19, -8
; ELF-NEXT:  .cfi_offset w30, -16
; ELF-NEXT: [[PRECALL:.L.*]]:
; ELF-NEXT:  blraaz x0

; ELF-SDAG-NEXT: [[POSTCALL:.L.*]]:
; ELF-SDAG-NEXT: // %bb.1:
; ELF-SDAG-NEXT:  mov w19, w0

; ELF-GISEL-NEXT:  mov w19, w0
; ELF-GISEL-NEXT: [[POSTCALL:.L.*]]:

; ELF-NEXT: [[CALLBB:.L.*]]:
; ELF-NEXT:  bl foo
; ELF-NEXT:  mov w0, w19
; ELF-NEXT:  ldp x30, x19, [sp], #16
; ELF-NEXT:  ret
; ELF-NEXT: [[LPADBB:.LBB[0-9_]+]]:
; ELF-NEXT: [[LPAD:.L.*]]:
; ELF-NEXT:  mov w19, #-1
; ELF-NEXT:  b [[CALLBB]]

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK:       .uleb128 [[POSTCALL]]-[[PRECALL]] {{.*}} Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:  .uleb128 [[LPAD]]-[[FNBEGIN]]     {{.*}}   jumps to [[LPAD]]
; CHECK-NEXT:  .byte 0                           {{.*}} On action: cleanup

define i32 @test_invoke_ia_0(ptr %arg0) #0 personality ptr @__gxx_personality_v0 {
  %tmp0 = invoke i32 %arg0() [ "ptrauth"(i32 0, i64 0) ] to label %continuebb
            unwind label %unwindbb

unwindbb:
  %tmp1 = landingpad { ptr, i32 } cleanup
  call void @foo()
  ret i32 -1

continuebb:
  call void @foo()
  ret i32 %tmp0
}

@_ZTIPKc = external constant ptr
@hello_str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

; DARWIN-LABEL: _test_invoke_ib_42_catch:
; DARWIN-NEXT: [[FNBEGIN:L.*]]:
; DARWIN-NEXT:         .cfi_startproc
; DARWIN-NEXT:         .cfi_personality 155, ___gxx_personality_v0
; DARWIN-NEXT:         .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; DARWIN-NEXT: ; %bb.0:
; DARWIN-NEXT:         stp x20, x19, [sp, #-32]!
; DARWIN-NEXT:         stp x29, x30, [sp, #16]
; DARWIN-NEXT:         .cfi_def_cfa_offset 32
; DARWIN-NEXT:         .cfi_offset w30, -8
; DARWIN-NEXT:         .cfi_offset w29, -16
; DARWIN-NEXT:         .cfi_offset w19, -24
; DARWIN-NEXT:         .cfi_offset w20, -32
; DARWIN-NEXT:         mov x19, x0
; DARWIN-NEXT:         mov w0, #8
; DARWIN-NEXT:         bl ___cxa_allocate_exception
; DARWIN-NEXT: Lloh{{.*}}:
; DARWIN-NEXT:         adrp x8, l_hello_str@PAGE
; DARWIN-NEXT: Lloh{{.*}}:
; DARWIN-NEXT:         add x8, x8, l_hello_str@PAGEOFF
; DARWIN-NEXT:         str x8, [x0]
; DARWIN-NEXT: [[PRECALL:L.*]]:
; DARWIN-NEXT: Lloh{{.*}}:
; DARWIN-NEXT:         adrp x1, __ZTIPKc@GOTPAGE
; DARWIN-NEXT: Lloh{{.*}}:
; DARWIN-NEXT:         ldr x1, [x1, __ZTIPKc@GOTPAGEOFF]
; DARWIN-NEXT:         mov x2, #0
; DARWIN-NEXT:         mov x17, #42
; DARWIN-NEXT:         blrab x19, x17
; DARWIN-NEXT: [[POSTCALL:L.*]]:
; DARWIN-NEXT: ; %bb.1:
; DARWIN-NEXT:         brk #0x1
; DARWIN-NEXT: [[LPADBB:LBB[0-9_]+]]:
; DARWIN-NEXT: [[LPAD:L.*]]:
; DARWIN-NEXT:         mov x19, x1
; DARWIN-NEXT:         bl ___cxa_begin_catch
; DARWIN-NEXT:         cmp     w19, #2
; DARWIN-NEXT:         b.ne [[EXITBB:LBB[0-9_]+]]
; DARWIN-NEXT: ; %bb.3:
; DARWIN-NEXT:         bl _bar
; DARWIN-NEXT: [[EXITBB]]:
; DARWIN-NEXT:         bl _foo
; DARWIN-NEXT:         bl ___cxa_end_catch
; DARWIN-NEXT:         ldp x29, x30, [sp, #16]
; DARWIN-NEXT:         ldp x20, x19, [sp], #32
; DARWIN-NEXT:         ret
; DARWIN-NEXT:         .loh {{.*}}
; DARWIN-NEXT:         .loh {{.*}}
; DARWIN-NEXT: [[FNEND:L.*]]:

; ELF-LABEL: test_invoke_ib_42_catch:
; ELF-NEXT: [[FNBEGIN:.L.*]]:
; ELF-NEXT:         .cfi_startproc
; ELF-NEXT:         .cfi_personality 156, DW.ref.__gxx_personality_v0
; ELF-NEXT:         .cfi_lsda 28, [[EXCEPT:.Lexception[0-9]+]]
; ELF-NEXT: // %bb.0:
; ELF-NEXT:         stp x30, x19, [sp, #-16]!
; ELF-NEXT:         .cfi_def_cfa_offset 16
; ELF-NEXT:         .cfi_offset w19, -8
; ELF-NEXT:         .cfi_offset w30, -16
; ELF-NEXT:         mov x19, x0
; ELF-NEXT:         mov w0, #8
; ELF-NEXT:         bl __cxa_allocate_exception
; ELF-NEXT:         adrp x8, .Lhello_str
; ELF-NEXT:         add x8, x8, :lo12:.Lhello_str
; ELF-NEXT:         str x8, [x0]
; ELF-NEXT: [[PRECALL:.L.*]]:
; ELF-NEXT:         adrp x1, :got:_ZTIPKc
; ELF-NEXT:         mov x2, xzr
; ELF-NEXT:         ldr x1, [x1, :got_lo12:_ZTIPKc]
; ELF-NEXT:         mov x17, #42
; ELF-NEXT:         blrab x19, x17
; ELF-NEXT: [[POSTCALL:.L.*]]:
; ELF-NEXT: // %bb.1:
; ELF-NEXT: [[LPADBB:.LBB[0-9_]+]]:
; ELF-NEXT: [[LPAD:.L.*]]:
; ELF-NEXT:         mov x19, x1
; ELF-NEXT:         bl __cxa_begin_catch
; ELF-NEXT:         cmp     w19, #2
; ELF-NEXT:         b.ne [[EXITBB:.LBB[0-9_]+]]
; ELF-NEXT: // %bb.3:
; ELF-NEXT:         bl bar
; ELF-NEXT: [[EXITBB]]:
; ELF-NEXT:         bl foo
; ELF-NEXT:         bl __cxa_end_catch
; ELF-NEXT:         ldp x30, x19, [sp], #16
; ELF-NEXT:         ret
; ELF-NEXT: [[FNEND:.L.*]]:

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK-NEXT:   .byte   255                       {{.*}} @LPStart Encoding = omit
; DARWIN-NEXT:  .byte   155                       {{.*}} @TType Encoding = indirect pcrel sdata4
; ELF-NEXT:     .byte   156                       {{.*}} @TType Encoding = indirect pcrel sdata8
; CHECK-NEXT:   .uleb128 [[TT:.?L.*]]-[[TTREF:.?L.*]]
; CHECK-NEXT: [[TTREF]]:
; CHECK-NEXT:   .byte   1                         {{.*}} Call site Encoding = uleb128
; CHECK-NEXT:   .uleb128 [[CSEND:.?L.*]]-[[CSBEGIN:.?L.*]]
; CHECK-NEXT: [[CSBEGIN]]:
; CHECK-NEXT:   .uleb128 [[FNBEGIN]]-[[FNBEGIN]]  {{.*}} >> Call Site 1 <<
; CHECK-NEXT:   .uleb128 [[PRECALL]]-[[FNBEGIN]]  {{.*}}   Call between [[FNBEGIN]] and [[PRECALL]]
; CHECK-NEXT:   .byte   0                         {{.*}}     has no landing pad
; CHECK-NEXT:   .byte   0                         {{.*}}   On action: cleanup
; CHECK-NEXT:   .uleb128 [[PRECALL]]-[[FNBEGIN]]  {{.*}} >> Call Site 2 <<
; CHECK-NEXT:   .uleb128 [[POSTCALL]]-[[PRECALL]] {{.*}}   Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:   .uleb128 [[LPAD]]-[[FNBEGIN]]     {{.*}}     jumps to [[LPAD]]
; CHECK-NEXT:   .byte   3                         {{.*}}   On action: 2
; CHECK-NEXT:   .uleb128 [[POSTCALL]]-[[FNBEGIN]] {{.*}} >> Call Site 3 <<
; CHECK-NEXT:   .uleb128 [[FNEND]]-[[POSTCALL]]   {{.*}}   Call between [[POSTCALL]] and [[FNEND]]
; CHECK-NEXT:   .byte   0                         {{.*}}     has no landing pad
; CHECK-NEXT:   .byte   0                         {{.*}}   On action: cleanup
; CHECK-NEXT: [[CSEND]]:

; CHECK-NEXT:   .byte   1                         {{.*}} >> Action Record 1 <<
; CHECK-NEXT:                                     {{.*}}   Catch TypeInfo 1
; CHECK-NEXT:   .byte   0                         {{.*}}   No further actions
; CHECK-NEXT:   .byte   2                         {{.*}} >> Action Record 2 <<
; CHECK-NEXT:                                     {{.*}}   Catch TypeInfo 2
; CHECK-NEXT:   .byte   125                       {{.*}}   Continue to action 1
; CHECK-NEXT:   .p2align   2
; CHECK-NEXT:                                     {{.*}} >> Catch TypeInfos <<

; DARWIN-NEXT: [[TI:L.*]]:                        {{.*}} TypeInfo 2
; DARWIN-NEXT:  .long   __ZTIPKc@GOT-[[TI]]
; DARWIN-NEXT:  .long   0                         {{.*}} TypeInfo 1
; ELF-NEXT:    [[TI:.?L.*]]:                      {{.*}} TypeInfo 2
; ELF-NEXT:     .xword  .L_ZTIPKc.DW.stub-[[TI]]
; ELF-NEXT:     .xword   0                        {{.*}} TypeInfo 1

; CHECK-NEXT: [[TT]]:

define void @test_invoke_ib_42_catch(ptr %fptr) #0 personality ptr @__gxx_personality_v0 {
  %tmp0 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds ([6 x i8], ptr @hello_str, i64 0, i64 0), ptr %tmp0, align 8
  invoke void %fptr(ptr %tmp0, ptr @_ZTIPKc, ptr null) [ "ptrauth"(i32 1, i64 42) ]
          to label %continuebb unwind label %catchbb

catchbb:
  %tmp2 = landingpad { ptr, i32 }
          catch ptr @_ZTIPKc
          catch ptr null
  %tmp3 = extractvalue { ptr, i32 } %tmp2, 0
  %tmp4 = extractvalue { ptr, i32 } %tmp2, 1
  %tmp5 = call i32 @llvm.eh.typeid.for(ptr @_ZTIPKc)
  %tmp6 = icmp eq i32 %tmp4, %tmp5
  %tmp7 = call ptr @__cxa_begin_catch(ptr %tmp3)
  br i1 %tmp6, label %PKc_catchbb, label %any_catchbb

PKc_catchbb:
  call void @bar(ptr %tmp7)
  br label %any_catchbb

any_catchbb:
  call void @foo()
  call void @__cxa_end_catch()
  ret void

continuebb:
  unreachable
}

; DARWIN-LABEL: _test_invoke_ia_0_direct:
; DARWIN-NEXT: [[FNBEGIN:L.*]]:
; DARWIN-NEXT:  .cfi_startproc
; DARWIN-NEXT:  .cfi_personality 155, ___gxx_personality_v0
; DARWIN-NEXT:  .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; DARWIN-NEXT: ; %bb.0:
; DARWIN-NEXT:  stp x20, x19, [sp, #-32]!
; DARWIN-NEXT:  stp x29, x30, [sp, #16]
; DARWIN-NEXT:  .cfi_def_cfa_offset 32
; DARWIN-NEXT:  .cfi_offset w30, -8
; DARWIN-NEXT:  .cfi_offset w29, -16
; DARWIN-NEXT:  .cfi_offset w19, -24
; DARWIN-NEXT:  .cfi_offset w20, -32
; DARWIN-NEXT: [[PRECALL:L.*]]:
; DARWIN-NEXT:  bl _baz

; DARWIN-SDAG-NEXT: [[POSTCALL:L.*]]:
; DARWIN-SDAG-NEXT: ; %bb.1:
; DARWIN-SDAG-NEXT:  mov x19, x0

; DARWIN-GISEL-NEXT:  mov x19, x0
; DARWIN-GISEL-NEXT: [[POSTCALL:L.*]]:

; DARWIN-NEXT: [[CALLBB:L.*]]:
; DARWIN-NEXT:  bl _foo
; DARWIN-NEXT:  mov x0, x19
; DARWIN-NEXT:  ldp x29, x30, [sp, #16]
; DARWIN-NEXT:  ldp x20, x19, [sp], #32
; DARWIN-NEXT:  ret
; DARWIN-NEXT: [[LPADBB:LBB[0-9_]+]]:
; DARWIN-NEXT: [[LPAD:L.*]]:
; DARWIN-NEXT:  mov w19, #-1
; DARWIN-NEXT:  b [[CALLBB]]

; ELF-LABEL: test_invoke_ia_0_direct:
; ELF-NEXT: [[FNBEGIN:.L.*]]:
; ELF-NEXT:  .cfi_startproc
; ELF-NEXT:  .cfi_personality 156, DW.ref.__gxx_personality_v0
; ELF-NEXT:  .cfi_lsda 28, [[EXCEPT:.Lexception[0-9]+]]
; ELF-NEXT: // %bb.0:
; ELF-NEXT:  stp x30, x19, [sp, #-16]!
; ELF-NEXT:  .cfi_def_cfa_offset 16
; ELF-NEXT:  .cfi_offset w19, -8
; ELF-NEXT:  .cfi_offset w30, -16
; ELF-NEXT: [[PRECALL:.L.*]]:
; ELF-NEXT:  bl baz

; ELF-SDAG-NEXT: [[POSTCALL:.L.*]]:
; ELF-SDAG-NEXT: // %bb.1:
; ELF-SDAG-NEXT:  mov w19, w0

; ELF-GISEL-NEXT:  mov w19, w0
; ELF-GISEL-NEXT: [[POSTCALL:.L.*]]:

; ELF-NEXT: [[CALLBB:.L.*]]:
; ELF-NEXT:  bl foo
; ELF-NEXT:  mov w0, w19
; ELF-NEXT:  ldp x30, x19, [sp], #16
; ELF-NEXT:  ret
; ELF-NEXT: [[LPADBB:.LBB[0-9_]+]]:
; ELF-NEXT: [[LPAD:.L.*]]:
; ELF-NEXT:  mov w19, #-1
; ELF-NEXT:  b [[CALLBB]]

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK:       .uleb128 [[POSTCALL]]-[[PRECALL]] {{.*}} Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:  .uleb128 [[LPAD]]-[[FNBEGIN]]     {{.*}}   jumps to [[LPAD]]
; CHECK-NEXT:  .byte 0                           {{.*}} On action: cleanup

define i32 @test_invoke_ia_0_direct() #0 personality ptr @__gxx_personality_v0 {
  %tmp0 = invoke i32 ptrauth (ptr @baz, i32 0)() [ "ptrauth"(i32 0, i64 0) ] to label %continuebb
            unwind label %unwindbb

unwindbb:
  %tmp1 = landingpad { ptr, i32 } cleanup
  call void @foo()
  ret i32 -1

continuebb:
  call void @foo()
  ret i32 %tmp0
}

; DARWIN-LABEL: _test_invoke_ib_2_direct_mismatch:
; DARWIN-NEXT: [[FNBEGIN:L.*]]:
; DARWIN-NEXT:  .cfi_startproc
; DARWIN-NEXT:  .cfi_personality 155, ___gxx_personality_v0
; DARWIN-NEXT:  .cfi_lsda 16, [[EXCEPT:Lexception[0-9]+]]
; DARWIN-NEXT: ; %bb.0:
; DARWIN-NEXT:  stp x20, x19, [sp, #-32]!
; DARWIN-NEXT:  stp x29, x30, [sp, #16]
; DARWIN-NEXT:  .cfi_def_cfa_offset 32
; DARWIN-NEXT:  .cfi_offset w30, -8
; DARWIN-NEXT:  .cfi_offset w29, -16
; DARWIN-NEXT:  .cfi_offset w19, -24
; DARWIN-NEXT:  .cfi_offset w20, -32

; DARWIN-SDAG-NEXT: [[PRECALL:L.*]]:
; DARWIN-SDAG-NEXT:  adrp x16, _baz@GOTPAGE
; DARWIN-SDAG-NEXT:  ldr x16, [x16, _baz@GOTPAGEOFF]
; DARWIN-SDAG-NEXT:  mov x17, #1234
; DARWIN-SDAG-NEXT:  pacia x16, x17
; DARWIN-SDAG-NEXT:  mov x8, x16
; DARWIN-SDAG-NEXT:  mov x17, #2
; DARWIN-SDAG-NEXT:  blrab x8, x17
; DARWIN-SDAG-NEXT: [[POSTCALL:L.*]]:
; DARWIN-SDAG-NEXT: ; %bb.1:
; DARWIN-SDAG-NEXT:  mov x19, x0

; DARWIN-GISEL-NEXT:  adrp x16, _baz@GOTPAGE
; DARWIN-GISEL-NEXT:  ldr x16, [x16, _baz@GOTPAGEOFF]
; DARWIN-GISEL-NEXT:  mov x17, #1234
; DARWIN-GISEL-NEXT:  pacia x16, x17
; DARWIN-GISEL-NEXT:  mov x8, x16
; DARWIN-GISEL-NEXT: [[PRECALL:L.*]]:
; DARWIN-GISEL-NEXT:  mov x17, #2
; DARWIN-GISEL-NEXT:  blrab x8, x17
; DARWIN-GISEL-NEXT:  mov x19, x0
; DARWIN-GISEL-NEXT: [[POSTCALL:L.*]]:

; DARWIN-NEXT: [[CALLBB:L.*]]:
; DARWIN-NEXT:  bl _foo
; DARWIN-NEXT:  mov x0, x19
; DARWIN-NEXT:  ldp x29, x30, [sp, #16]
; DARWIN-NEXT:  ldp x20, x19, [sp], #32
; DARWIN-NEXT:  ret
; DARWIN-NEXT: [[LPADBB:LBB[0-9_]+]]:
; DARWIN-NEXT: [[LPAD:L.*]]:
; DARWIN-NEXT:  mov w19, #-1
; DARWIN-NEXT:  b [[CALLBB]]

; ELF-LABEL: test_invoke_ib_2_direct_mismatch:
; ELF-NEXT: [[FNBEGIN:.L.*]]:
; ELF-NEXT:  .cfi_startproc
; ELF-NEXT:  .cfi_personality 156, DW.ref.__gxx_personality_v0
; ELF-NEXT:  .cfi_lsda 28, [[EXCEPT:.Lexception[0-9]+]]
; ELF-NEXT: // %bb.0:
; ELF-NEXT:  stp x30, x19, [sp, #-16]!
; ELF-NEXT:  .cfi_def_cfa_offset 16
; ELF-NEXT:  .cfi_offset w19, -8
; ELF-NEXT:  .cfi_offset w30, -16

; ELF-SDAG-NEXT: [[PRECALL:.L.*]]:
; ELF-SDAG-NEXT:  adrp x16, :got:baz
; ELF-SDAG-NEXT:  ldr x16, [x16, :got_lo12:baz]
; ELF-SDAG-NEXT:  mov x17, #1234
; ELF-SDAG-NEXT:  pacia x16, x17
; ELF-SDAG-NEXT:  mov x8, x16
; ELF-SDAG-NEXT:  mov x17, #2
; ELF-SDAG-NEXT:  blrab x8, x17
; ELF-SDAG-NEXT: [[POSTCALL:.L.*]]:
; ELF-SDAG-NEXT: // %bb.1:
; ELF-SDAG-NEXT:  mov w19, w0

; ELF-GISEL-NEXT:  adrp x16, :got:baz
; ELF-GISEL-NEXT:  ldr x16, [x16, :got_lo12:baz]
; ELF-GISEL-NEXT:  mov x17, #1234
; ELF-GISEL-NEXT:  pacia x16, x17
; ELF-GISEL-NEXT:  mov x8, x16
; ELF-GISEL-NEXT: [[PRECALL:.L.*]]:
; ELF-GISEL-NEXT:  mov x17, #2
; ELF-GISEL-NEXT:  blrab x8, x17
; ELF-GISEL-NEXT:  mov w19, w0
; ELF-GISEL-NEXT: [[POSTCALL:.L.*]]:

; ELF-NEXT: [[CALLBB:.L.*]]:
; ELF-NEXT:  bl foo
; ELF-NEXT:  mov w0, w19
; ELF-NEXT:  ldp x30, x19, [sp], #16
; ELF-NEXT:  ret
; ELF-NEXT: [[LPADBB:.LBB[0-9_]+]]:
; ELF-NEXT: [[LPAD:.L.*]]:
; ELF-NEXT:  mov w19, #-1
; ELF-NEXT:  b [[CALLBB]]

; CHECK-LABEL: GCC_except_table{{.*}}:
; CHECK-NEXT: [[EXCEPT]]:
; CHECK:       .uleb128 [[POSTCALL]]-[[PRECALL]] {{.*}} Call between [[PRECALL]] and [[POSTCALL]]
; CHECK-NEXT:  .uleb128 [[LPAD]]-[[FNBEGIN]]     {{.*}}   jumps to [[LPAD]]
; CHECK-NEXT:  .byte 0                           {{.*}} On action: cleanup

define i32 @test_invoke_ib_2_direct_mismatch() #0 personality ptr @__gxx_personality_v0 {
  %tmp0 = invoke i32 ptrauth (ptr @baz, i32 0, i64 1234)() [ "ptrauth"(i32 1, i64 2) ] to label %continuebb
            unwind label %unwindbb

unwindbb:
  %tmp1 = landingpad { ptr, i32 } cleanup
  call void @foo()
  ret i32 -1

continuebb:
  call void @foo()
  ret i32 %tmp0
}

; ELF-LABEL:  .L_ZTIPKc.DW.stub:
; ELF-NEXT:     .xword  _ZTIPKc

declare void @foo()
declare void @bar(ptr)
declare i32 @baz()

declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr)
declare i32 @llvm.eh.typeid.for(ptr)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

attributes #0 = { nounwind }
