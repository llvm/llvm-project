; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare hhvmcc i64 @bar(i64, i64, i64) nounwind

; Simply check we can modify %rbx and %rbp before returning via call to bar.
define hhvmcc i64 @foo(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  foo:
; CHECK-DAG:    movl $1, %ebx
; CHECK-DAG:    movl $3, %ebp
; CHECK:        jmp bar
  %ret = musttail call hhvmcc i64 @bar(i64 1, i64 %b, i64 3)
  ret i64 %ret
}

; Check that we can read and modify %rbx returned from PHP function.
define hhvmcc i64 @mod_return(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  mod_return:
; CHECK-NEXT:   {{^#.*}}
; CHECK-NEXT:   callq bar
; CHECK-NEXT:   incq %rbx
  %tmp = call hhvmcc i64 @bar(i64 %a, i64 %b, i64 %c)
  %retval = add i64 %tmp, 1
  ret i64 %retval
}

%rettype = type { i64, i64, i64, i64, i64, i64, i64,
                  i64, i64, i64, i64, i64, i64, i64
}

; Check that we can return up to 14 64-bit args in registers.
define hhvmcc %rettype @return_all(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  return_all:
; CHECK-DAG:    movl $1, %ebx
; CHECK-DAG:    movl $2, %ebp
; CHECK-DAG:    movl $3, %edi
; CHECK-DAG:    movl $4, %esi
; CHECK-DAG:    movl $5, %edx
; CHECK-DAG:    movl $6, %ecx
; CHECK-DAG:    movl $7, %r8
; CHECK-DAG:    movl $8, %r9
; CHECK-DAG:    movl $9, %eax
; CHECK-DAG:    movl $10, %r10
; CHECK-DAG:    movl $11, %r11
; CHECK-DAG:    movl $12, %r13
; CHECK-DAG:    movl $13, %r14
; CHECK-DAG:    movl $14, %r15
; CHECK:        retq
  %r1 = insertvalue %rettype zeroinitializer, i64 1, 0
  %r2 = insertvalue %rettype %r1, i64 2, 1
  %r3 = insertvalue %rettype %r2, i64 3, 2
  %r4 = insertvalue %rettype %r3, i64 4, 3
  %r5 = insertvalue %rettype %r4, i64 5, 4
  %r6 = insertvalue %rettype %r5, i64 6, 5
  %r7 = insertvalue %rettype %r6, i64 7, 6
  %r8 = insertvalue %rettype %r7, i64 8, 7
  %r9 = insertvalue %rettype %r8, i64 9, 8
  %r10 = insertvalue %rettype %r9, i64 10, 9
  %r11 = insertvalue %rettype %r10, i64 11, 10
  %r12 = insertvalue %rettype %r11, i64 12, 11
  %r13 = insertvalue %rettype %r12, i64 13, 12
  %r14 = insertvalue %rettype %r13, i64 14, 13
  ret %rettype %r14
}

declare hhvmcc void @return_all_tc(i64, i64, i64, i64, i64, i64, i64, i64,
                                 i64, i64, i64, i64, i64, i64, i64)

; Check that we can return up to 14 64-bit args in registers via tail call.
define hhvmcc void @test_return_all_tc(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_return_all_tc:
; CHECK-NEXT:   {{^#.*}}
; CHECK-DAG:    movl $1, %ebx
; CHECK-DAG:    movl $3, %ebp
; CHECK-DAG:    movl $4, %r15
; CHECK-DAG:    movl $5, %edi
; CHECK-DAG:    movl $6, %esi
; CHECK-DAG:    movl $7, %edx
; CHECK-DAG:    movl $8, %ecx
; CHECK-DAG:    movl $9, %r8
; CHECK-DAG:    movl $10, %r9
; CHECK-DAG:    movl $11, %eax
; CHECK-DAG:    movl $12, %r10
; CHECK-DAG:    movl $13, %r11
; CHECK-DAG:    movl $14, %r13
; CHECK-DAG:    movl $15, %r14
; CHECK:        jmp  return_all_tc
  tail call hhvmcc void @return_all_tc(
    i64 1, i64 %b, i64 3, i64 4, i64 5, i64 6, i64 7,
    i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15)
  ret void
}

declare hhvmcc {i64, i64} @php_short(i64, i64, i64, i64)

define hhvmcc i64 @test_php_short(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_php_short:
; CHECK-NEXT:   {{^#.*}}
; CHECK-NEXT:   movl $42, %r15
; CHECK-NEXT:   callq php_short
; CHECK-NEXT:   leaq (%rbp,%r12), %rbx
; CHECK-NEXT:   retq
  %pair = call hhvmcc {i64, i64} @php_short(i64 %a, i64 %b, i64 %c, i64 42)
  %fp = extractvalue {i64, i64} %pair, 1
  %rv = add i64 %fp, %b
  ret i64 %rv
}

declare hhvmcc %rettype @php_all(i64, i64, i64, i64, i64, i64, i64,
                                 i64, i64, i64, i64, i64, i64, i64, i64)

; Check that we can pass 15 arguments in registers.
; Also check that %r12 (2nd arg) is not spilled.
define hhvmcc i64 @test_php_all(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_php_all:
; CHECK-NEXT:   {{^#.*}}
; CHECK-NOT:    sub
; CHECK-NOT:    sub
; CHECK-DAG:    movl $1, %ebx
; CHECK-DAG:    movl $3, %ebp
; CHECK-DAG:    movl $4, %r15
; CHECK-DAG:    movl $5, %edi
; CHECK-DAG:    movl $6, %esi
; CHECK-DAG:    movl $7, %edx
; CHECK-DAG:    movl $8, %ecx
; CHECK-DAG:    movl $9, %r8
; CHECK-DAG:    movl $10, %r9
; CHECK-DAG:    movl $11, %eax
; CHECK-DAG:    movl $12, %r10
; CHECK-DAG:    movl $13, %r11
; CHECK-DAG:    movl $14, %r13
; CHECK-DAG:    movl $15, %r14
; CHECK:        callq php_all
  %pair = call hhvmcc %rettype @php_all(
    i64 1, i64 %b, i64 3, i64 4, i64 5, i64 6, i64 7,
    i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15)
  %fp = extractvalue %rettype %pair, 1
  %rv = add i64 %fp, %b
  ret i64 %rv
}

declare hhvmcc void @svcreq(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
                             i64, i64)

define hhvmcc void @test_svcreq(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_svcreq:
; CHECK-DAG:    movl $42, %r10
; CHECK-DAG:    movl $1, %edi
; CHECK-DAG:    movl $2, %esi
; CHECK-DAG:    movl $3, %edx
; CHECK-DAG:    movl $4, %ecx
; CHECK-DAG:    movl $5, %r8
; CHECK-DAG:    movl $6, %r9
; CHECK:        jmp svcreq
  tail call hhvmcc void @svcreq(i64 %a, i64 %b, i64 %c, i64 undef, i64 1,
                                i64 2, i64 3, i64 4, i64 5, i64 6, i64 undef,
                                i64 42)
  ret void
}

declare hhvm_ccc void @helper_short(i64, i64, i64, i64, i64, i64, i64)

; Pass all arguments in registers and check that we don't adjust stack
; for the call.
define hhvmcc void @test_helper_short(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_helper_short:
; CHECK-NOT:    push
; CHECK-NOT:    sub
; CHECK-DAG:    movl $1, %edi
; CHECK-DAG:    movl $2, %esi
; CHECK-DAG:    movl $3, %edx
; CHECK-DAG:    movl $4, %ecx
; CHECK-DAG:    movl $5, %r8
; CHECK-DAG:    movl $6, %r9
; CHECK:        callq helper_short
  call hhvm_ccc void @helper_short(i64 %c, i64 1, i64 2, i64 3, i64 4,
                                   i64 5, i64 6)
  ret void
}

declare hhvm_ccc void @helper(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define hhvmcc void @test_helper(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_helper:
; CHECK-DAG:    movl $1, %edi
; CHECK-DAG:    movl $2, %esi
; CHECK-DAG:    movl $3, %edx
; CHECK-DAG:    movl $4, %ecx
; CHECK-DAG:    movl $5, %r8
; CHECK-DAG:    movl $6, %r9
; CHECK:        callq helper
  call hhvm_ccc void @helper(i64 %c, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6,
                             i64 7, i64 8, i64 9)
  ret void
}

; When we enter function with HHVM calling convention, the stack is aligned
; at 16 bytes. This means we align objects on the stack differently and
; adjust the stack differently for calls.
declare hhvm_ccc void @stack_helper(i64, i64, i64)
declare hhvm_ccc void @stack_helper2(<2 x double>, i64)

define hhvmcc void @test_stack_helper(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_stack_helper:
; CHECK-NOT:    push
; CHECK:        subq $32, %rsp
; CHECK:        movaps  16(%rsp), %xmm0
; CHECK:        callq stack_helper2
  %t1 = alloca <2 x double>, align 16
  %t2 = alloca i64, align 8
  %t3 = alloca i64, align 8
  %load3 = load i64, ptr%t3
  call hhvm_ccc void @stack_helper(i64 %c, i64 %load3, i64 42)
  %load = load <2 x double>, ptr%t1
  %load2 = load i64, ptr%t2
  call hhvm_ccc void @stack_helper2(<2 x double> %load, i64 %load2)
  ret void
}

; Check that we are not adjusting the stack before calling the helper.
define hhvmcc void @test_stack_helper2(i64 %a, i64 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL:  test_stack_helper2:
; CHECK-NOT:    push
; CHECK-NOT:    subq
  call hhvm_ccc void @stack_helper(i64 %c, i64 7, i64 42)
  ret void
}

