; RUN: llc < %s -stack-symbol-ordering=0 -tailcallopt -code-model=medium -mtriple=i686-linux-gnu -mcpu=pentium | FileCheck %s

; Check the HiPE calling convention works (x86-32)

define void @zap(i32 %a, i32 %b) nounwind {
entry:
  ; CHECK:      movl 40(%esp), %eax
  ; CHECK-NEXT: movl 44(%esp), %edx
  ; CHECK-NEXT: movl       $8, %ecx
  ; CHECK-NEXT: calll addfour
  %0 = call cc 11 {i32, i32, i32} @addfour(i32 undef, i32 undef, i32 %a, i32 %b, i32 8)
  %res = extractvalue {i32, i32, i32} %0, 2

  ; CHECK:      movl %eax, 16(%esp)
  ; CHECK-NEXT: movl   $2, 12(%esp)
  ; CHECK-NEXT: movl   $1,  8(%esp)
  ; CHECK:      calll foo
  tail call void @foo(i32 undef, i32 undef, i32 1, i32 2, i32 %res) nounwind
  ret void
}

define cc 11 {i32, i32, i32} @addfour(i32 %hp, i32 %p, i32 %x, i32 %y, i32 %z) nounwind {
entry:
  ; CHECK:      addl %edx, %eax
  ; CHECK-NEXT: addl %ecx, %eax
  %0 = add i32 %x, %y
  %1 = add i32 %0, %z

  ; CHECK:      ret
  %res = insertvalue {i32, i32, i32} undef, i32 %1, 2
  ret {i32, i32, i32} %res
}

define cc 11 void @foo(i32 %hp, i32 %p, i32 %arg0, i32 %arg1, i32 %arg2) nounwind {
entry:
  ; CHECK:      movl  %esi, 16(%esp)
  ; CHECK-NEXT: movl  %ebp, 12(%esp)
  ; CHECK-NEXT: movl  %eax,  8(%esp)
  ; CHECK-NEXT: movl  %edx,  4(%esp)
  ; CHECK-NEXT: movl  %ecx,   (%esp)
  %hp_var   = alloca i32
  %p_var    = alloca i32
  %arg0_var = alloca i32
  %arg1_var = alloca i32
  %arg2_var = alloca i32
  store i32 %hp, i32* %hp_var
  store i32 %p, i32* %p_var
  store i32 %arg0, i32* %arg0_var
  store i32 %arg1, i32* %arg1_var
  store i32 %arg2, i32* %arg2_var
  ; These loads are loading the values from their previous stores and are optimized away.
  %0 = load i32, i32* %hp_var
  %1 = load i32, i32* %p_var
  %2 = load i32, i32* %arg0_var
  %3 = load i32, i32* %arg1_var
  %4 = load i32, i32* %arg2_var
  ; CHECK:      jmp bar
  tail call cc 11 void @bar(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4) nounwind
  ret void
}

define cc 11 void @baz() nounwind {
  %tmp_clos = load i32, i32* @clos
  %tmp_clos2 = inttoptr i32 %tmp_clos to i32*
  %indirect_call = bitcast i32* %tmp_clos2 to void (i32, i32, i32)*
  ; CHECK:      movl $42, %eax
  ; CHECK-NEXT: jmpl *clos
  tail call cc 11 void %indirect_call(i32 undef, i32 undef, i32 42) nounwind
  ret void
}

; Sanity-check the tail call sequence. Number of arguments was chosen as to
; expose a bug where the tail call sequence clobbered the stack.
define cc 11 { i32, i32, i32 } @tailcaller(i32 %hp, i32 %p) nounwind {
  ; CHECK:      movl	$15, %eax
  ; CHECK-NEXT: movl	$31, %edx
  ; CHECK-NEXT: movl	$47, %ecx
  ; CHECK-NEXT: popl	%edi
  ; CHECK-NEXT: jmp	tailcallee
  %ret = tail call cc11 { i32, i32, i32 } @tailcallee(i32 %hp, i32 %p, i32 15,
     i32 31, i32 47, i32 63) nounwind
  ret { i32, i32, i32 } %ret
}

!hipe.literals = !{ !0, !1, !2 }
!0 = !{ !"P_NSP_LIMIT", i32 84 }
!1 = !{ !"X86_LEAF_WORDS", i32 24 }
!2 = !{ !"AMD64_LEAF_WORDS", i32 24 }
@clos = external dso_local constant i32
declare cc 11 void @bar(i32, i32, i32, i32, i32)
declare cc 11 { i32, i32, i32 } @tailcallee(i32, i32, i32, i32, i32, i32)

!llvm.module.flags = !{!3}
!3 = !{i32 2, !"override-stack-alignment", i32 4}
