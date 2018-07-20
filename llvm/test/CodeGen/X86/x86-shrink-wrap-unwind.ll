; RUN: llc %s -o - | FileCheck %s
;
; Note: This test cannot be merged with the shrink-wrapping tests
; because the booleans set on the command line take precedence on
; the target logic that disable shrink-wrapping.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-apple-macosx"


; This test checks that we do not use shrink-wrapping when
; the function does not have any frame pointer and may unwind.
; This is a workaround for a limitation in the emission of
; the CFI directives, that are not correct in such case.
; PR25614
;
; No shrink-wrapping should occur here, until the CFI information are fixed.
; CHECK-LABEL: framelessUnwind:
;
; Prologue code.
; (What we push does not matter. It should be some random sratch register.)
; CHECK: pushq
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; CHECK: movl %edi, [[ARG0CPY:%e[a-z]+]]
; CHECK-NEXT: cmpl %esi, %edi
; CHECK-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], 4(%rsp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq 4(%rsp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; CHECK: [[EXIT_LABEL]]:
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; CHECK-NEXT: popq
;
; CHECK-NEXT: retq
define i32 @framelessUnwind(i32 %a, i32 %b) #0 {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

declare i32 @doSomething(i32, i32*)

attributes #0 = { "no-frame-pointer-elim"="false" }

; Shrink-wrapping should occur here. We have a frame pointer.
; CHECK-LABEL: frameUnwind:
;
; Compare the arguments and jump to exit.
; No prologue needed.
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; CHECK: movl %edi, [[ARG0CPY:%e[a-z]+]]
; CHECK-NEXT: cmpl %esi, %edi
; CHECK-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], -4(%rbp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq -4(%rbp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; Epilogue code. (What we pop does not matter.)
; CHECK: popq %rbp
;
; CHECK: [[EXIT_LABEL]]:
; CHECK-NEXT: retq
define i32 @frameUnwind(i32 %a, i32 %b) #1 {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

attributes #1 = { "no-frame-pointer-elim"="true" }

; Shrink-wrapping should occur here. We do not have to unwind.
; CHECK-LABEL: framelessnoUnwind:
;
; Compare the arguments and jump to exit.
; No prologue needed.
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; CHECK: movl %edi, [[ARG0CPY:%e[a-z]+]]
; CHECK-NEXT: cmpl %esi, %edi
; CHECK-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; (What we push does not matter. It should be some random sratch register.)
; CHECK: pushq
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], 4(%rsp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq 4(%rsp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; Epilogue code.
; CHECK-NEXT: addq
;
; CHECK: [[EXIT_LABEL]]:
; CHECK-NEXT: retq
define i32 @framelessnoUnwind(i32 %a, i32 %b) #2 {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

attributes #2 = { "no-frame-pointer-elim"="false" nounwind }


; Check that we generate correct code for segmented stack.
; We used to emit the code at the entry point of the function
; instead of just before the prologue.
; For now, shrink-wrapping is disabled on segmented stack functions: PR26107.
;
; CHECK-LABEL: segmentedStack:
; CHECK: cmpq
; CHECK-NEXT: ja [[ENTRY_LABEL:LBB[0-9_]+]]
;
; CHECK: callq ___morestack
; CHECK-NEXT: retq
;
; CHECK: [[ENTRY_LABEL]]:
; Prologue
; CHECK: push
;
; In PR26107, we use to drop these two basic blocks, because
; the segmentedStack entry block was jumping directly to
; the place where the prologue is actually needed, which is
; the call to memcmp.
; Then, those two basic blocks did not have any predecessors
; anymore and were removed.
;
; Check if vk1 is null
; CHECK: testq %rdi, %rdi
; CHECK-NEXT: je [[STRINGS_EQUAL:LBB[0-9_]+]]
;
; Check if vk2 is null
; CHECK: testq %rsi, %rsi
; CHECK-NEXT:  je [[STRINGS_EQUAL]]
;
; CHECK: [[STRINGS_EQUAL]]
; CHECK: popq
define zeroext i1 @segmentedStack(i8* readonly %vk1, i8* readonly %vk2, i64 %key_size) #5 {
entry:
  %cmp.i = icmp eq i8* %vk1, null
  %cmp1.i = icmp eq i8* %vk2, null
  %brmerge.i = or i1 %cmp.i, %cmp1.i
  %cmp1.mux.i = and i1 %cmp.i, %cmp1.i
  br i1 %brmerge.i, label %__go_ptr_strings_equal.exit, label %if.end4.i

if.end4.i:                                        ; preds = %entry
  %tmp = getelementptr inbounds i8, i8* %vk1, i64 8
  %tmp1 = bitcast i8* %tmp to i64*
  %tmp2 = load i64, i64* %tmp1, align 8
  %tmp3 = getelementptr inbounds i8, i8* %vk2, i64 8
  %tmp4 = bitcast i8* %tmp3 to i64*
  %tmp5 = load i64, i64* %tmp4, align 8
  %cmp.i.i = icmp eq i64 %tmp2, %tmp5
  br i1 %cmp.i.i, label %land.rhs.i.i, label %__go_ptr_strings_equal.exit

land.rhs.i.i:                                     ; preds = %if.end4.i
  %tmp6 = bitcast i8* %vk2 to i8**
  %tmp7 = load i8*, i8** %tmp6, align 8
  %tmp8 = bitcast i8* %vk1 to i8**
  %tmp9 = load i8*, i8** %tmp8, align 8
  %call.i.i = tail call i32 @memcmp(i8* %tmp9, i8* %tmp7, i64 %tmp2) #5
  %cmp4.i.i = icmp eq i32 %call.i.i, 0
  br label %__go_ptr_strings_equal.exit

__go_ptr_strings_equal.exit:                      ; preds = %land.rhs.i.i, %if.end4.i, %entry
  %retval.0.i = phi i1 [ %cmp1.mux.i, %entry ], [ false, %if.end4.i ], [ %cmp4.i.i, %land.rhs.i.i ]
  ret i1 %retval.0.i
}

; Function Attrs: nounwind readonly
declare i32 @memcmp(i8* nocapture, i8* nocapture, i64) #5

attributes #5 = { nounwind readonly ssp uwtable "split-stack" }

; Check that correctly take into account the jumps to landing pad.
; We used to consider function that may throw like regular
; function calls.
; Therefore, in this example, we were happily inserting the epilogue
; right after the call to throw_exception. Because of that we would not
; execute the epilogue when an execption occur and bad things will
; happen.
; PR36513
;
; CHECK-LABEL: with_nounwind:
; Prologue
; CHECK: push
;
; Jump to throw_exception:
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: jne [[THROW_LABEL:LBB[0-9_]+]]
; Else return exit
; CHECK: popq
; CHECK-NEXT: retq
;
; CHECK-NEXT: [[THROW_LABEL]]:
; CHECK: callq	_throw_exception
; Unreachable block...
;
; Epilogue must be after the landing pad.
; CHECK-NOT: popq
;
; Look for the landing pad label.
; CHECK: LBB{{[0-9_]+}}:
; Epilogue on the landing pad
; CHECK: popq
; CHECK-NEXT: retq
define void @with_nounwind(i1 %cond) nounwind personality i32 (...)* @my_personality {
entry:
  br i1 %cond, label %throw, label %return

throw:
  invoke void @throw_exception()
          to label %unreachable unwind label %landing

unreachable:
  unreachable

landing:
  %pad = landingpad { i8*, i32 }
          catch i8* null
  ret void

return:
  ret void
}

; Check landing pad again.
; This time checks that we can shrink-wrap when the epilogue does not
; span accross several blocks.
;
; CHECK-LABEL: with_nounwind_same_succ:
;
; Jump to throw_exception:
; CHECK: testb $1, %dil
; CHECK-NEXT: je [[RET_LABEL:LBB[0-9_]+]]
;
; Prologue
; CHECK: push
; CHECK: callq	_throw_exception
;
; Fallthrough label
; CHECK: [[FALLTHROUGH_LABEL:LBB[0-9_]+]]
; CHECK: nop
; CHECK: popq
;
; CHECK: [[RET_LABEL]]
; CHECK: retq
;
; Look for the landing pad label.
; CHECK: LBB{{[0-9_]+}}:
; Landing pad jumps to fallthrough
; CHECK: jmp [[FALLTHROUGH_LABEL]]
define void @with_nounwind_same_succ(i1 %cond) nounwind personality i32 (...)* @my_personality2 {
entry:
  br i1 %cond, label %throw, label %return

throw:
  invoke void @throw_exception()
          to label %fallthrough unwind label %landing
landing:
  %pad = landingpad { i8*, i32 }
          catch i8* null
  br label %fallthrough

fallthrough:
  tail call void asm "nop", ""()
  br label %return

return:
  ret void
}

declare void @throw_exception()
declare i32 @my_personality(...)
declare i32 @my_personality2(...)
