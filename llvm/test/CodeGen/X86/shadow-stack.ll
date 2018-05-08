; RUN: llc -mtriple x86_64-unknown-unknown -mattr=+shstk < %s | FileCheck %s --check-prefix=X86_64
; RUN: llc -mtriple i386-unknown-unknown -mattr=+shstk < %s | FileCheck %s --check-prefix=X86

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; The IR was created using the following C code:
;; typedef void *jmp_buf;
;; jmp_buf *buf;
;;
;; __attribute__((noinline)) int bar (int i) {
;;   int j = i - 111;
;;   __builtin_longjmp (buf, 1);
;;   return j;
;; }
;;
;; int foo (int i) {
;;   int j = i * 11;
;;   if (!__builtin_setjmp (buf)) {
;;     j += 33 + bar (j);
;;   }
;;   return j + i;
;; }
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

@buf = common local_unnamed_addr global i8* null, align 8

; Functions that use LongJmp should fix the Shadow Stack using previosuly saved
; ShadowStackPointer in the input buffer.
; The fix requires unwinding the shadow stack to the last SSP.
define i32 @bar(i32 %i) local_unnamed_addr {
; X86_64-LABEL: bar:
; X86_64:         movq {{.*}}(%rip), %rax
; X86_64-NEXT:    xorq %rdx, %rdx
; X86_64-NEXT:    rdsspq %rdx
; X86_64-NEXT:    testq %rdx, %rdx
; X86_64-NEXT:    je .LBB0_5
; X86_64-NEXT:  # %bb.1: # %entry
; X86_64-NEXT:    movq 24(%rax), %rcx
; X86_64-NEXT:    subq %rdx, %rcx
; X86_64-NEXT:    jbe .LBB0_5
; X86_64-NEXT:  # %bb.2: # %entry
; X86_64-NEXT:    shrq $3, %rcx
; X86_64-NEXT:    incsspq %rcx
; X86_64-NEXT:    shrq $8, %rcx
; X86_64-NEXT:    je .LBB0_5
; X86_64-NEXT:  # %bb.3: # %entry
; X86_64-NEXT:    shlq %rcx
; X86_64-NEXT:    movq $128, %rdx
; X86_64-NEXT:  .LBB0_4: # %entry
; X86_64-NEXT:    # =>This Inner Loop Header: Depth=1
; X86_64-NEXT:    incsspq %rdx
; X86_64-NEXT:    decq %rcx
; X86_64-NEXT:    jne .LBB0_4
; X86_64-NEXT:  .LBB0_5: # %entry
; X86_64-NEXT:    movq (%rax), %rbp
; X86_64-NEXT:    movq 8(%rax), %rcx
; X86_64-NEXT:    movq 16(%rax), %rsp
; X86_64-NEXT:    jmpq *%rcx
;
; X86-LABEL: bar:
; X86:         movl buf, %eax
; X86-NEXT:    xorl %edx, %edx
; X86-NEXT:    rdsspd %edx
; X86-NEXT:    testl %edx, %edx
; X86-NEXT:    je .LBB0_5
; X86-NEXT:  # %bb.1: # %entry
; X86-NEXT:    movl 12(%eax), %ecx
; X86-NEXT:    subl %edx, %ecx
; X86-NEXT:    jbe .LBB0_5
; X86-NEXT:  # %bb.2: # %entry
; X86-NEXT:    shrl $2, %ecx
; X86-NEXT:    incsspd %ecx
; X86-NEXT:    shrl $8, %ecx
; X86-NEXT:    je .LBB0_5
; X86-NEXT:  # %bb.3: # %entry
; X86-NEXT:    shll %ecx
; X86-NEXT:    movl $128, %edx
; X86-NEXT:  .LBB0_4: # %entry
; X86-NEXT:    # =>This Inner Loop Header: Depth=1
; X86-NEXT:    incsspd %edx
; X86-NEXT:    decl %ecx
; X86-NEXT:    jne .LBB0_4
; X86-NEXT:  .LBB0_5: # %entry
; X86-NEXT:    movl (%eax), %ebp
; X86-NEXT:    movl 4(%eax), %ecx
; X86-NEXT:    movl 8(%eax), %esp
; X86-NEXT:    jmpl *%ecx
entry:
  %0 = load i8*, i8** @buf, align 8
  tail call void @llvm.eh.sjlj.longjmp(i8* %0)
  unreachable
}

declare void @llvm.eh.sjlj.longjmp(i8*)

; Functions that call SetJmp should save the current ShadowStackPointer for
; future fixing of the Shadow Stack.
define i32 @foo(i32 %i) local_unnamed_addr {
; X86_64-LABEL: foo:
; X86_64:         xorq %rcx, %rcx
; X86_64-NEXT:    rdsspq %rcx
; X86_64-NEXT:    movq %rcx, 24(%rax)
; X86_64:         callq bar
;
; X86-LABEL: foo:
; X86:         xorl %ecx, %ecx
; X86-NEXT:    rdsspd %ecx
; X86-NEXT:    movl %ecx, 12(%eax)
; X86:         calll bar
entry:
  %0 = load i8*, i8** @buf, align 8
  %1 = bitcast i8* %0 to i8**
  %2 = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %2, i8** %1, align 8
  %3 = tail call i8* @llvm.stacksave()
  %4 = getelementptr inbounds i8, i8* %0, i64 16
  %5 = bitcast i8* %4 to i8**
  store i8* %3, i8** %5, align 8
  %6 = tail call i32 @llvm.eh.sjlj.setjmp(i8* %0)
  %tobool = icmp eq i32 %6, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = tail call i32 @bar(i32 undef)
  unreachable

if.end:                                           ; preds = %entry
  %add2 = mul nsw i32 %i, 12
  ret i32 %add2
}

declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.stacksave()
declare i32 @llvm.eh.sjlj.setjmp(i8*)
