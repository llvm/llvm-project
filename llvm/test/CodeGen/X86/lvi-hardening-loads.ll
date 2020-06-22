; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown < %s | FileCheck %s --check-prefix=X64 --check-prefix=X64-ALL
; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown --x86-lvi-load-no-cbranch < %s | FileCheck %s --check-prefix=X64
; RUN: llc -O0 -verify-machineinstrs -mtriple=x86_64-unknown < %s | FileCheck %s --check-prefix=X64-NOOPT

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @test(i32** %secret, i32 %secret_size) #0 {
; X64-LABEL: test:
entry:
  %secret.addr = alloca i32**, align 8
  %secret_size.addr = alloca i32, align 4
  %ret_val = alloca i32, align 4
  %i = alloca i32, align 4
  store i32** %secret, i32*** %secret.addr, align 8
  store i32 %secret_size, i32* %secret_size.addr, align 4
  store i32 0, i32* %ret_val, align 4
  call void @llvm.x86.sse2.lfence()
  store i32 0, i32* %i, align 4
  br label %for.cond

; X64: # %bb.0: # %entry
; X64-NEXT:      movq %rdi, -{{[0-9]+}}(%rsp)
; X64-NEXT:      movl %esi, -{{[0-9]+}}(%rsp)
; X64-NEXT:      movl $0, -{{[0-9]+}}(%rsp)
; X64-NEXT:      lfence
; X64-NEXT:      movl $0, -{{[0-9]+}}(%rsp)
; X64-NEXT:      jmp .LBB0_1

; X64-NOOPT: # %bb.0: # %entry
; X64-NOOPT-NEXT:      movq %rdi, -{{[0-9]+}}(%rsp)
; X64-NOOPT-NEXT:      movl %esi, -{{[0-9]+}}(%rsp)
; X64-NOOPT-NEXT:      movl $0, -{{[0-9]+}}(%rsp)
; X64-NOOPT-NEXT:      lfence
; X64-NOOPT-NEXT:      movl $0, -{{[0-9]+}}(%rsp)

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %secret_size.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

; X64: .LBB0_1: # %for.cond
; X64-NEXT:      # =>This Inner Loop Header: Depth=1
; X64-NEXT:      movl -{{[0-9]+}}(%rsp), %eax
; X64-ALL-NEXT:  lfence
; X64-NEXT:      cmpl -{{[0-9]+}}(%rsp), %eax
; X64-ALL-NEXT:  lfence
; X64-NEXT:      jge .LBB0_5

; X64-NOOPT: .LBB0_1: # %for.cond
; X64-NOOPT-NEXT:      # =>This Inner Loop Header: Depth=1
; X64-NOOPT-NEXT:      movl -{{[0-9]+}}(%rsp), %eax
; X64-NOOPT-NEXT:  lfence
; X64-NOOPT-NEXT:      cmpl -{{[0-9]+}}(%rsp), %eax
; X64-NOOPT-NEXT:  lfence
; X64-NOOPT-NEXT:      jge .LBB0_6

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %rem = srem i32 %2, 2
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.end

; X64: # %bb.2: # %for.body
; X64-NEXT: # in Loop: Header=BB0_1 Depth=1
; X64-NEXT:      movl -{{[0-9]+}}(%rsp), %eax
; X64-ALL-NEXT:  lfence
; X64-NEXT:      movl %eax, %ecx
; X64-NEXT:      shrl $31, %ecx
; X64-NEXT:      addl %eax, %ecx
; X64-NEXT:      andl $-2, %ecx
; X64-NEXT:      cmpl %ecx, %eax
; X64-NEXT:      jne .LBB0_4

; X64-NOOPT: # %bb.2: # %for.body
; X64-NOOPT-NEXT: # in Loop: Header=BB0_1 Depth=1
; X64-NOOPT-NEXT:      movl -{{[0-9]+}}(%rsp), %eax
; X64-NOOPT-NEXT:  lfence
; X64-NOOPT-NEXT:      cltd
; X64-NOOPT-NEXT:      movl $2, %ecx
; X64-NOOPT-NEXT:      idivl %ecx
; X64-NOOPT-NEXT:      cmpl $0, %edx
; X64-NOOPT-NEXT:      jne .LBB0_4

if.then:                                          ; preds = %for.body
  %3 = load i32**, i32*** %secret.addr, align 8
  %4 = load i32, i32* %ret_val, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds i32*, i32** %3, i64 %idxprom
  %5 = load i32*, i32** %arrayidx, align 8
  %6 = load i32, i32* %5, align 4
  store i32 %6, i32* %ret_val, align 4
  br label %if.end

; X64: # %bb.3: # %if.then
; X64-NEXT: # in Loop: Header=BB0_1 Depth=1
; X64-NEXT:      movq -{{[0-9]+}}(%rsp), %rax
; X64-NEXT:      lfence
; X64-NEXT:      movslq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:      lfence
; X64-NEXT:      movq (%rax,%rcx,8), %rax
; X64-NEXT:      lfence
; X64-NEXT:      movl (%rax), %eax
; X64-NEXT:      movl %eax, -{{[0-9]+}}(%rsp)
; X64-NEXT:      jmp .LBB0_4

; X64-NOOPT: # %bb.3: # %if.then
; X64-NOOPT-NEXT: # in Loop: Header=BB0_1 Depth=1
; X64-NOOPT-NEXT:      movq -{{[0-9]+}}(%rsp), %rax
; X64-NOOPT-NEXT:      lfence
; X64-NOOPT-NEXT:      movslq -{{[0-9]+}}(%rsp), %rcx
; X64-NOOPT-NEXT:      lfence
; X64-NOOPT-NEXT:      movq (%rax,%rcx,8), %rax
; X64-NOOPT-NEXT:      lfence
; X64-NOOPT-NEXT:      movl (%rax), %eax
; X64-NOOPT-NEXT:      lfence
; X64-NOOPT-NEXT:      movl %eax, -{{[0-9]+}}(%rsp)

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

; X64-NOOPT: .LBB0_5: # %for.inc
; X64-NOOPT-NEXT: # in Loop: Header=BB0_1 Depth=1
; X64-NOOPT-NEXT:      movl -{{[0-9]+}}(%rsp), %eax
; X64-NOOPT-NEXT:      lfence
; X64-NOOPT-NEXT:      addl $1, %eax
; X64-NOOPT-NEXT:      movl %eax, -{{[0-9]+}}(%rsp)
; X64-NOOPT-NEXT:      jmp .LBB0_1

for.end:                                          ; preds = %for.cond
  %8 = load i32, i32* %ret_val, align 4
  ret i32 %8
}

; Function Attrs: nounwind
declare void @llvm.x86.sse2.lfence() #1

attributes #0 = { "target-features"="+lvi-load-hardening" }
attributes #1 = { nounwind }
