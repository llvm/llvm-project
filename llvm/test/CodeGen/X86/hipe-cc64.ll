; RUN: llc < %s -stack-symbol-ordering=0 -tailcallopt -relocation-model=static -code-model=medium -mtriple=x86_64-linux-gnu -mcpu=opteron | FileCheck %s

; Check the HiPE calling convention works (x86-64)

define void @zap(i64 %a, i64 %b) nounwind {
entry:
  ; CHECK:      movq %rsi, %rdx
  ; CHECK-NEXT: movl $8, %ecx
  ; CHECK-NEXT: movl $9, %r8d
  ; CHECK-NEXT: movq %rdi, %rsi
  ; CHECK-NEXT: callq addfour
  %0 = call cc 11 {i64, i64, i64} @addfour(i64 undef, i64 undef, i64 %a, i64 %b, i64 8, i64 9)
  %res = extractvalue {i64, i64, i64} %0, 2

  ; CHECK:      movl $1, %edx
  ; CHECK-NEXT: movl $2, %ecx
  ; CHECK-NEXT: movl $3, %r8d
  ; CHECK-NEXT: movq %rax, %r9
  ; CHECK:      callq foo
  tail call void @foo(i64 undef, i64 undef, i64 1, i64 2, i64 3, i64 %res) nounwind
  ret void
}

define cc 11 {i64, i64, i64} @addfour(i64 %hp, i64 %p, i64 %x, i64 %y, i64 %z, i64 %w) nounwind {
entry:
  ; CHECK:      leaq (%rdx,%rcx), %rax
  ; CHECK-NEXT: addq %r8, %rax
  ; CHECK-NEXT: addq %rsi, %rax
  %0 = add i64 %x, %y
  %1 = add i64 %0, %z
  %2 = add i64 %1, %w

  ; CHECK:      ret
  %res = insertvalue {i64, i64, i64} undef, i64 %2, 2
  ret {i64, i64, i64} %res
}

define cc 11 void @foo(i64 %hp, i64 %p, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3) nounwind {
entry:
  ; CHECK:      movq  %r15, 40(%rsp)
  ; CHECK-NEXT: movq  %rbp, 32(%rsp)
  ; CHECK-NEXT: movq  %rsi, 24(%rsp)
  ; CHECK-NEXT: movq  %rdx, 16(%rsp)
  ; CHECK-NEXT: movq  %rcx, 8(%rsp)
  ; CHECK-NEXT: movq  %r8, (%rsp)
  %hp_var   = alloca i64
  %p_var    = alloca i64
  %arg0_var = alloca i64
  %arg1_var = alloca i64
  %arg2_var = alloca i64
  %arg3_var = alloca i64
  store i64 %hp, ptr %hp_var
  store i64 %p, ptr %p_var
  store i64 %arg0, ptr %arg0_var
  store i64 %arg1, ptr %arg1_var
  store i64 %arg2, ptr %arg2_var
  store i64 %arg3, ptr %arg3_var

  ; Loads are reading values just writen from corresponding register and are therefore noops. 
  %0 = load i64, ptr %hp_var
  %1 = load i64, ptr %p_var
  %2 = load i64, ptr %arg0_var
  %3 = load i64, ptr %arg1_var
  %4 = load i64, ptr %arg2_var
  %5 = load i64, ptr %arg3_var
  ; CHECK:      jmp bar
  tail call cc 11 void @bar(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5) nounwind
  ret void
}

define cc 11 void @baz() nounwind {
  %tmp_clos = load i64, ptr @clos
  %tmp_clos2 = inttoptr i64 %tmp_clos to ptr
  ; CHECK:      movl $42, %esi
  ; CHECK-NEXT: jmpq *(%rax)
  tail call cc 11 void %tmp_clos2(i64 undef, i64 undef, i64 42) nounwind
  ret void
}

; Sanity-check the tail call sequence. Number of arguments was chosen as to
; expose a bug where the tail call sequence clobbered the stack.
define cc 11 { i64, i64, i64 } @tailcaller(i64 %hp, i64 %p) #0 {
  ; CHECK:      movl	$15, %esi
  ; CHECK-NEXT: movl	$31, %edx
  ; CHECK-NEXT: movl	$47, %ecx
  ; CHECK-NEXT: movl	$63, %r8d
  ; CHECK-NEXT: popq	%rax
  ; CHECK-NEXT: .cfi_def_cfa_offset 16
  ; CHECK-NEXT: jmp	tailcallee
  %ret = tail call cc11 { i64, i64, i64 } @tailcallee(i64 %hp, i64 %p, i64 15,
     i64 31, i64 47, i64 63, i64 79) #1
  ret { i64, i64, i64 } %ret
}

!hipe.literals = !{ !0, !1, !2 }
!0 = !{ !"P_NSP_LIMIT", i32 160 }
!1 = !{ !"X86_LEAF_WORDS", i32 24 }
!2 = !{ !"AMD64_LEAF_WORDS", i32 24 }
@clos = external constant i64
declare cc 11 void @bar(i64, i64, i64, i64, i64, i64)
declare cc 11 { i64, i64, i64 } @tailcallee(i64, i64, i64, i64, i64, i64, i64)
!llvm.module.flags = !{!3}
!3 = !{i32 2, !"override-stack-alignment", i32 8}
