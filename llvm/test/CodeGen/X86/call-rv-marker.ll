; RUN: llc -mtriple=x86_64-apple-macosx -verify-machineinstrs -o - %s | FileCheck --check-prefix=CHECK %s

; TODO: support marker generation with GlobalISel
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

declare i8* @foo0(i32)
declare i8* @foo1()

declare void @llvm.objc.release(i8*)
declare void @objc_object(i8*)

declare void @foo2(i8*)

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare %struct.S* @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1))

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)


%struct.S = type { i8 }

@g = global i8* null, align 8
@fptr = global i8* ()* null, align 8

define i8* @rv_marker_1_retain() {
; CHECK-LABEL:  rv_marker_1_retain:
; CHECK:         pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    callq   _foo1
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:    popq    %rcx
; CHECK-NEXT:    retq
;
entry:
  %call = call i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  ret i8* %call
}

define i8* @rv_marker_1_claim() {
; CHECK-LABEL:  rv_marker_1_claim:
; CHECK:         pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    callq   _foo1
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _objc_unsafeClaimAutoreleasedReturnValue
; CHECK-NEXT:    popq    %rcx
; CHECK-NEXT:    retq
;
entry:
  %call = call i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_unsafeClaimAutoreleasedReturnValue) ]
  ret i8* %call
}

define void @rv_marker_2_select(i32 %c) {
; CHECK-LABEL: rv_marker_2_select:
; CHECK:        pushq   %rax
; CHECK-NEXT:   .cfi_def_cfa_offset 16
; CHECK-NEXT:   cmpl    $1, %edi
; CHECK-NEXT:   movl    $1, %edi
; CHECK-NEXT:   adcl    $0, %edi
; CHECK-NEXT:   callq   _foo0
; CHECK-NEXT:   movq    %rax, %rdi
; CHECK-NEXT:   callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:   movq    %rax, %rdi
; CHECK-NEXT:   popq    %rax
; CHECK-NEXT:   jmp _foo2
;
entry:
  %tobool.not = icmp eq i32 %c, 0
  %.sink = select i1 %tobool.not, i32 2, i32 1
  %call1 = call i8* @foo0(i32 %.sink) [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call1)
  ret void
}

define void @rv_marker_3() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: rv_marker_3
; CHECK:         pushq   %r14
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    pushq   %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 24
; CHECK-NEXT:    pushq   %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    .cfi_offset %rbx, -24
; CHECK-NEXT:    .cfi_offset %r14, -16
; CHECK-NEXT:    callq   _foo1
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:    movq    %rax, %rbx
; CHECK-NEXT: Ltmp0:
;
entry:
  %call = call i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  invoke void @objc_object(i8* %call) #5
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  tail call void @llvm.objc.release(i8* %call)
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          cleanup
  tail call void @llvm.objc.release(i8* %call)
  resume { i8*, i32 } %0
}

define void @rv_marker_4() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: rv_marker_4
; CHECK:         pushq   %r14
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    pushq   %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 24
; CHECK-NEXT:    pushq   %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    .cfi_offset %rbx, -24
; CHECK-NEXT:    .cfi_offset %r14, -16
; CHECK-NEXT: Ltmp3:
; CHECK-NEXT:    callq   _foo1
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT: Ltmp4:
;
entry:
  %s = alloca %struct.S, align 1
  %0 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0) #2
  %call = invoke i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  invoke void @objc_object(i8* %call) #5
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %invoke.cont
  tail call void @llvm.objc.release(i8* %call)
  %call3 = call %struct.S* @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1) %s)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0)
  ret void

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

lpad1:                                            ; preds = %invoke.cont
  %2 = landingpad { i8*, i32 }
          cleanup
  tail call void @llvm.objc.release(i8* %call)
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad1, %lpad
  %.pn = phi { i8*, i32 } [ %2, %lpad1 ], [ %1, %lpad ]
  %call4 = call %struct.S* @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1) %s)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0)
  resume { i8*, i32 } %.pn
}

; TODO: This should use "callq *_fptr(%rip)".
define i8* @rv_marker_5_indirect_call() {
; CHECK-LABEL: rv_marker_5_indirect_call
; CHECK:         pushq   %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset %rbx, -16
; CHECK-NEXT:    movq    _fptr(%rip), %rax
; CHECK-NEXT:    callq   *%rax
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:    movq    %rax, %rbx
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _foo2
; CHECK-NEXT:    movq    %rbx, %rax
; CHECK-NEXT:    popq    %rbx
; CHECK-NEXT:    retq
;
entry:
  %lv = load i8* ()*, i8* ()** @fptr, align 8
  %call = call i8* %lv() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call)
  ret i8* %call
}

declare i8* @foo(i64, i64, i64)

define void @rv_marker_multiarg(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: rv_marker_multiarg
; CHECK:         pushq   %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    movq    %rdi, %rax
; CHECK-NEXT:    movq    %rdx, %rdi
; CHECK-NEXT:    movq    %rax, %rdx
; CHECK-NEXT:    callq   _foo
; CHECK-NEXT:    movq    %rax, %rdi
; CHECK-NEXT:    callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:    popq    %rax
; CHECK-NEXT:    retq
;
  %r = call i8* @foo(i64 %c, i64 %b, i64 %a) [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  ret void
}

define void @test_nonlazybind() {
; CHECK-LABEL: _test_nonlazybind:
; CHECK:      bb.0:
; CHECK-NEXT:  pushq   %rax
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  callq   *_foo_nonlazybind@GOTPCREL(%rip)
; CHECK-NEXT:  movq    %rax, %rdi
; CHECK-NEXT:  callq   _objc_retainAutoreleasedReturnValue
;
  %call1 = notail call i8* @foo_nonlazybind() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  ret void
}

declare i8* @foo_nonlazybind()  nonlazybind

declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_unsafeClaimAutoreleasedReturnValue(i8*)
declare i32 @__gxx_personality_v0(...)

declare i8* @fn1()
declare i8* @fn2()

define i8* @rv_marker_block_placement(i1 %c.0) {
; CHECK-LABEL: _rv_marker_block_placement:
; CHECK:        pushq   %rax
; CHECK-NEXT:   .cfi_def_cfa_offset 16
; CHECK-NEXT:   testb   $1, %dil
; CHECK-NEXT:   je  LBB8_2

; CHECK-NEXT: ## %bb.1:
; CHECK-NEXT:   callq   _fn1
; CHECK-NEXT:   movq    %rax, %rdi
; CHECK-NEXT:   callq   _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:   jmp LBB8_3

; CHECK-NEXT: LBB8_2:
; CHECK-NEXT:   callq   _fn2
; CHECK-NEXT:   movq    %rax, %rdi
; CHECK-NEXT:   callq   _objc_retainAutoreleasedReturnValue

; CHECK-NEXT: LBB8_3:
; CHECK-NEXT:   xorl    %eax, %eax
; CHECK-NEXT:   popq    %rcx
; CHECK-NEXT:   retq
;
entry:
  br i1 %c.0, label %then, label %else

then:
  %call.0 = notail call i8* @fn1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  br label %exit

else:
  %call.1 = notail call i8* @fn2() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  br label %exit

exit:
  ret i8* null
}
