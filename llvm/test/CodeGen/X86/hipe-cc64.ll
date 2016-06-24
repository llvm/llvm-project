; RUN: llc < %s -stack-symbol-ordering=0 -tailcallopt -code-model=medium -stack-alignment=8 -mtriple=x86_64-linux-gnu -mcpu=opteron | FileCheck %s

; Check the HiPE calling convention works (x86-64)

define void @zap(i64 %a, i64 %b) nounwind {
entry:
  ; CHECK:      movq %rsi, %rax
  ; CHECK-NEXT: movl $8, %ecx
  ; CHECK-NEXT: movl $9, %r8d
  ; CHECK-NEXT: movq %rdi, %rsi
  ; CHECK-NEXT: movq %rax, %rdx
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
  ; CHECK:      leaq (%rsi,%rdx), %rax
  ; CHECK-NEXT: addq %rcx, %rax
  ; CHECK-NEXT: addq %r8, %rax
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
  store i64 %hp, i64* %hp_var
  store i64 %p, i64* %p_var
  store i64 %arg0, i64* %arg0_var
  store i64 %arg1, i64* %arg1_var
  store i64 %arg2, i64* %arg2_var
  store i64 %arg3, i64* %arg3_var

  ; CHECK:      movq  40(%rsp), %r15
  ; CHECK-NEXT: movq  32(%rsp), %rbp
  ; CHECK-NEXT: movq  24(%rsp), %rsi
  ; CHECK-NEXT: movq  16(%rsp), %rdx
  ; CHECK-NEXT: movq  8(%rsp), %rcx
  %0 = load i64, i64* %hp_var
  %1 = load i64, i64* %p_var
  %2 = load i64, i64* %arg0_var
  %3 = load i64, i64* %arg1_var
  %4 = load i64, i64* %arg2_var
  %5 = load i64, i64* %arg3_var
  ; CHECK:      jmp bar
  tail call cc 11 void @bar(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5) nounwind
  ret void
}

define cc 11 void @baz() nounwind {
  %tmp_clos = load i64, i64* @clos
  %tmp_clos2 = inttoptr i64 %tmp_clos to i64*
  %indirect_call = bitcast i64* %tmp_clos2 to void (i64, i64, i64)*
  ; CHECK:      movl $42, %esi
  ; CHECK-NEXT: jmpq *(%rax)
  tail call cc 11 void %indirect_call(i64 undef, i64 undef, i64 42) nounwind
  ret void
}

!hipe.literals = !{ !0, !1, !2 }
!0 = !{ !"P_NSP_LIMIT", i32 160 }
!1 = !{ !"X86_LEAF_WORDS", i32 24 }
!2 = !{ !"AMD64_LEAF_WORDS", i32 24 }
@clos = external constant i64
declare cc 11 void @bar(i64, i64, i64, i64, i64, i64)
