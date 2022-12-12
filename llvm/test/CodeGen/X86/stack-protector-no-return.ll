; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -o - | FileCheck %s

$__clang_call_terminate = comdat any

@_ZTIi = external dso_local constant i8*
@.str = private unnamed_addr constant [5 x i8] c"win\0A\00", align 1

; Function Attrs: mustprogress noreturn sspreq uwtable
define dso_local void @_Z7catchesv() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: _Z7catchesv:
; CHECK:       # %bb.0: # %entry
; CHECK:         movq    %fs:40, %rax
; CHECK-NEXT:    movq    %rax, 8(%rsp)
entry:
  %exception = tail call i8* @__cxa_allocate_exception(i64 4) #8
  %0 = bitcast i8* %exception to i32*
  store i32 1, i32* %0, align 16
  invoke void @__cxa_throw(i8* nonnull %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #9
          to label %unreachable unwind label %lpad

; CHECK:         callq   __cxa_allocate_exception
; CHECK-NEXT:    movl    $1, (%rax)
; CHECK-NEXT:    movq    %fs:40, %rcx
; CHECK-NEXT:    cmpq    8(%rsp), %rcx
; CHECK-NEXT:    jne     .LBB0_12
; CHECK-NEXT:  # %bb.1:                                # %SP_return
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:    movl    $_ZTIi, %esi
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    xorl    %edx, %edx
; CHECK-NEXT:    callq   __cxa_throw

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2) #8
  %call = invoke i64 @write(i32 noundef 1, i8* noundef getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i64 noundef 4)
          to label %invoke.cont unwind label %lpad1
; CHECK:         callq   __cxa_begin_catch
; CHECK-NEXT: .Ltmp3:
; CHECK-NEXT:    movl    $.L.str, %esi
; CHECK-NEXT:    movl    $4, %edx
; CHECK-NEXT:    movl    $1, %edi
; CHECK-NEXT:    callq   write


invoke.cont:                                      ; preds = %lpad
  invoke void @_exit(i32 noundef 1) #9
          to label %invoke.cont2 unwind label %lpad1
; CHECK:       # %bb.3:                                # %invoke.cont
; CHECK-NEXT:    movq    %fs:40, %rax
; CHECK-NEXT:    cmpq    8(%rsp), %rax
; CHECK-NEXT:    jne     .LBB0_12
; CHECK-NEXT:  # %bb.4:                                # %SP_return3
; CHECK-NEXT:  .Ltmp5:
; CHECK-NEXT:    movl    $1, %edi
; CHECK-NEXT:    callq   _exit

invoke.cont2:                                     ; preds = %invoke.cont
  unreachable

lpad1:                                            ; preds = %invoke.cont, %lpad
  %4 = landingpad { i8*, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad
; CHECK: .LBB0_6:                                # %lpad1
; CHECK-NEXT: .Ltmp7:
; CHECK-NEXT:   movq    %rax, %rbx
; CHECK-NEXT: .Ltmp8:
; CHECK-NEXT:   callq   __cxa_end_catch

eh.resume:                                        ; preds = %lpad1
  resume { i8*, i32 } %4
; CHECK:      # %bb.7:                                # %eh.resume
; CHECK-NEXT:   movq    %fs:40, %rax
; CHECK-NEXT:   cmpq    8(%rsp), %rax
; CHECK-NEXT:   jne     .LBB0_12
; CHECK-NEXT: # %bb.8:                                # %SP_return6
; CHECK-NEXT:   movq    %rbx, %rdi
; CHECK-NEXT:   callq   _Unwind_Resume@PLT

terminate.lpad:                                   ; preds = %lpad1
  %5 = landingpad { i8*, i32 }
          catch i8* null
  %6 = extractvalue { i8*, i32 } %5, 0
  tail call void @__clang_call_terminate(i8* %6) #10
  unreachable

; CHECK: .LBB0_9:                                # %terminate.lpad
; CHECK:        movq    %fs:40, %rcx
; CHECK-NEXT:   cmpq    8(%rsp), %rcx
; CHECK-NEXT:   jne     .LBB0_12
; CHECK-NEXT: # %bb.10:                               # %SP_return9
; CHECK-NEXT:   movq    %rax, %rdi
; CHECK-NEXT:   callq   __clang_call_terminate

; CHECK:  .LBB0_12:                               # %CallStackCheckFailBlk
; CHECK-NEXT:   callq   __stack_chk_fail@PLT

unreachable:                                      ; preds = %entry
  unreachable
}

; Function Attrs: nofree
declare dso_local noalias i8* @__cxa_allocate_exception(i64) local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare dso_local void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr #2

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nofree
declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr #1

; Function Attrs: nofree
declare dso_local noundef i64 @write(i32 noundef, i8* nocapture noundef readonly, i64 noundef) local_unnamed_addr #3

; Function Attrs: nofree noreturn
declare dso_local void @_exit(i32 noundef) local_unnamed_addr #4

; Function Attrs: nofree
declare dso_local void @__cxa_end_catch() local_unnamed_addr #1

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8* %0) local_unnamed_addr #5 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #8
  tail call void @_ZSt9terminatev() #10
  unreachable
}

; Function Attrs: nofree noreturn nounwind
declare dso_local void @_ZSt9terminatev() local_unnamed_addr #6

; Function Attrs: mustprogress nofree sspreq uwtable
define dso_local void @_Z4vulni(i32 noundef %op) local_unnamed_addr #7 {
; CHECK-LABEL: _Z4vulni:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    movq %fs:40, %rax
; CHECK-NEXT:    movq %rax, (%rsp)
; CHECK-NEXT:    cmpl $1, %edi
; CHECK-NEXT:    je .LBB2_1
; CHECK-NEXT:  # %bb.3: # %if.end
; CHECK-NEXT:    movq %fs:40, %rax
; CHECK-NEXT:    cmpq (%rsp), %rax
; CHECK-NEXT:    jne .LBB2_5
; CHECK-NEXT:  # %bb.4: # %SP_return3
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
; CHECK-NEXT:  .LBB2_1: # %if.then
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    movl $4, %edi
; CHECK-NEXT:    callq __cxa_allocate_exception
; CHECK-NEXT:    movl $1, (%rax)
; CHECK-NEXT:    movq %fs:40, %rcx
; CHECK-NEXT:    cmpq (%rsp), %rcx
; CHECK-NEXT:    jne .LBB2_5
; CHECK-NEXT:  # %bb.2: # %SP_return
; CHECK-NEXT:    movl $_ZTIi, %esi
; CHECK-NEXT:    movq %rax, %rdi
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    callq __cxa_throw
; CHECK-NEXT:  .LBB2_5: # %CallStackCheckFailBlk2
; CHECK-NEXT:    callq __stack_chk_fail@PLT
entry:
  %cmp = icmp eq i32 %op, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %exception = tail call i8* @__cxa_allocate_exception(i64 4) #8
  %0 = bitcast i8* %exception to i32*
  store i32 1, i32* %0, align 16
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #9
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

attributes #0 = { mustprogress noreturn sspreq uwtable }
attributes #1 = { nofree }
attributes #2 = { nofree noreturn }
attributes #3 = { nofree }
attributes #4 = { nofree noreturn }
attributes #5 = { noinline noreturn nounwind }
attributes #6 = { nofree noreturn nounwind }
attributes #7 = { mustprogress nofree sspreq uwtable }
attributes #8 = { nounwind }
attributes #9 = { noreturn }
attributes #10 = { noreturn nounwind }
