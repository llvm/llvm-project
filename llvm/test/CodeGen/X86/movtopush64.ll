; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s -check-prefix=NORMAL -check-prefix=NORMALFP
; RUN: llc < %s -mtriple=x86_64-windows | FileCheck %s -check-prefix=NOPUSH
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s -check-prefix=NOPUSH -check-prefix=NORMALFP
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -no-x86-call-frame-opt | FileCheck %s -check-prefix=NOPUSH

declare void @seven_params(i32 %a, i64 %b, i32 %c, i64 %d, i32 %e, i64 %f, i32 %g)
declare void @eightparams(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h)
declare void @eightparams16(i16 %a, i16 %b, i16 %c, i16 %d, i16 %e, i16 %f, i16 %g, i16 %h)
declare void @eightparams64(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h)
declare void @ten_params(i32 %a, i64 %b, i32 %c, i64 %d, i32 %e, i64 %f, i32 %g, i64 %h, i32 %i, i64 %j)
declare void @ten_params_ptr(i32 %a, i64 %b, i32 %c, i64 %d, i32 %e, i64 %f, i32 %g, i8* %h, i32 %i, i64 %j)
declare void @cannot_push(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i)

; We should get pushes for the last 4 parameters. Test that the
; in-register parameters are all in the right places, and check
; that the stack manipulations are correct and correctly
; described by the DWARF directives. Test that the switch
; to disable the optimization works and that the optimization
; doesn't kick in on Windows64 where it is not allowed.
; NORMAL-LABEL: test1
; NORMAL: pushq
; NORMAL-DAG: movl $1, %edi
; NORMAL-DAG: movl $2, %esi
; NORMAL-DAG: movl $3, %edx
; NORMAL-DAG: movl $4, %ecx
; NORMAL-DAG: movl $5, %r8d
; NORMAL-DAG: movl $6, %r9d
; NORMAL: pushq $10
; NORMAL: .cfi_adjust_cfa_offset 8
; NORMAL: pushq $9
; NORMAL: .cfi_adjust_cfa_offset 8
; NORMAL: pushq $8
; NORMAL: .cfi_adjust_cfa_offset 8
; NORMAL: pushq $7
; NORMAL: .cfi_adjust_cfa_offset 8
; NORMAL: callq ten_params
; NORMAL: addq $32, %rsp
; NORMAL: .cfi_adjust_cfa_offset -32
; NORMAL: popq
; NORMAL: retq
; NOPUSH-LABEL: test1
; NOPUSH-NOT: pushq
; NOPUSH: retq
define void @test1() {
entry:
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i64 8, i32 9, i64 10)
  ret void
}

; The presence of a frame pointer should not prevent pushes. But we
; don't need the CFI directives in that case.
; Also check that we generate the right pushes for >8bit immediates.
; NORMALFP-LABEL: test2
; NORMALFP: pushq $10000
; NORMALFP-NEXT: pushq $9000
; NORMALFP-NEXT: pushq $8000
; NORMALFP-NEXT: pushq $7000
; NORMALFP-NEXT: callq {{_?}}ten_params
define void @test2(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7000, i64 8000, i32 9000, i64 10000)
  ret void
}

; Parameters 7 & 8 should push a 64-bit register.
; TODO: Note that the regular expressions disallow r8 and r9. That's fine for
;       now, because the pushes will always follow the moves into r8 and r9.
;       Eventually, though, we want to be able to schedule the pushes better.
;       In this example, it will save two copies, because we have to move the
;       incoming parameters out of %rdi and %rsi to make room for the outgoing
;       parameters.
; NORMAL-LABEL: test3
; NORMAL: pushq $10000
; NORMAL: pushq $9000
; NORMAL: pushq %r{{..}}
; NORMAL: pushq %r{{..}}
; NORMAL: callq ten_params
define void @test3(i32 %a, i64 %b) {
entry:
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 %a, i64 %b, i32 9000, i64 10000)
  ret void
}

; Check that we avoid the optimization for just one push.
; NORMAL-LABEL: test4
; NORMAL: movl $7, (%rsp)
; NORMAL: callq seven_params
define void @test4() {
entry:
  call void @seven_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7)
  ret void
}

; Check that pushing link-time constant addresses works correctly
; NORMAL-LABEL: test5
; NORMAL: pushq $10
; NORMAL: pushq $9
; NORMAL: pushq $ext
; NORMAL: pushq $7
; NORMAL: callq ten_params_ptr
@ext = external constant i8
define void @test5() {
entry:
  call void @ten_params_ptr(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i8* @ext, i32 9, i64 10)
  ret void
}

; Check that we fuse 64-bit loads but not 32-bit loads into PUSH mem.
; NORMAL-LABEL: test6
; NORMAL: movq %rsi, [[REG64:%.+]]
; NORMAL: pushq $10
; NORMAL: pushq $9
; NORMAL: pushq ([[REG64]])
; NORMAL: pushq {{%r..}}
; NORMAL: callq ten_params
define void @test6(i32* %p32, i64* %p64) {
entry:
  %v32 = load i32, i32* %p32
  %v64 = load i64, i64* %p64
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 %v32, i64 %v64, i32 9, i64 10)
  ret void
}

; Fold stack-relative loads into the push with correct offsets.
; Do the same for an indirect call whose address is loaded from the stack.
; On entry, %p7 is at 8(%rsp) and %p8 is at 16(%rsp). Prior to the call
; sequence, 72 bytes are allocated to the stack, 48 for register saves and
; 24 for local storage and alignment, so %p7 is at 80(%rsp) and %p8 is at
; 88(%rsp). The call address can be stored anywhere in the local space but
; happens to be stored at 8(%rsp). Each push bumps these offsets up by
; 8 bytes.
; NORMAL-LABEL: test7
; NORMAL: movq %r{{.*}}, 8(%rsp) {{.*Spill$}}
; NORMAL: pushq 88(%rsp)
; NORMAL: pushq $9
; NORMAL: pushq 96(%rsp)
; NORMAL: pushq $7
; NORMAL: callq *40(%rsp)
define void @test7(i64 %p1, i64 %p2, i64 %p3, i64 %p4, i64 %p5, i64 %p6, i64 %p7, i64 %p8) {
entry:
  %stack_fptr = alloca void (i32, i64, i32, i64, i32, i64, i32, i64, i32, i64)*
  store void (i32, i64, i32, i64, i32, i64, i32, i64, i32, i64)* @ten_params, void (i32, i64, i32, i64, i32, i64, i32, i64, i32, i64)** %stack_fptr
  %ten_params_ptr = load volatile void (i32, i64, i32, i64, i32, i64, i32, i64, i32, i64)*, void (i32, i64, i32, i64, i32, i64, i32, i64, i32, i64)** %stack_fptr
  call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  call void (i32, i64, i32, i64, i32, i64, i32, i64, i32, i64) %ten_params_ptr(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i64 %p7, i32 9, i64 %p8)
  ret void
}

; We can't fold the load from the global into the push because of 
; interference from the store
; NORMAL-LABEL: test8
; NORMAL: movq the_global(%rip), [[REG:%r.+]]
; NORMAL: movq $42, the_global
; NORMAL: pushq $10
; NORMAL: pushq $9
; NORMAL: pushq [[REG]]
; NORMAL: pushq $7
; NORMAL: callq ten_params
@the_global = external global i64
define void @test8() {
  %myload = load i64, i64* @the_global
  store i64 42, i64* @the_global
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i64 %myload, i32 9, i64 10)
  ret void
}


; Converting one function call to use pushes negatively affects
; other calls that pass arguments on the stack without pushes.
; If the cost outweighs the benefit, avoid using pushes.
; NORMAL-LABEL: test9
; NORMAL: callq cannot_push
; NORMAL-NOT: push
; NORMAL: callq ten_params
define void @test9(float %p1) {
  call void @cannot_push(float 1.0e0, float 2.0e0, float 3.0e0, float 4.0e0, float 5.0e0, float 6.0e0, float 7.0e0, float 8.0e0, float %p1)
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i64 8, i32 9, i64 10)
  call void @cannot_push(float 1.0e0, float 2.0e0, float 3.0e0, float 4.0e0, float 5.0e0, float 6.0e0, float 7.0e0, float 8.0e0, float %p1)
  ret void
}

; But if the benefit outweighs the cost, use pushes.
; NORMAL-LABEL: test10
; NORMAL: callq cannot_push
; NORMAL: pushq $10
; NORMAL: pushq $9
; NORMAL: pushq $8
; NORMAL: pushq $7
; NORMAL: callq ten_params
define void @test10(float %p1) {
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i64 8, i32 9, i64 10)
  call void @cannot_push(float 1.0e0, float 2.0e0, float 3.0e0, float 4.0e0, float 5.0e0, float 6.0e0, float 7.0e0, float 8.0e0, float %p1)
  call void @ten_params(i32 1, i64 2, i32 3, i64 4, i32 5, i64 6, i32 7, i64 8, i32 9, i64 10)
  ret void
}

; NORMAL-LABEL: pr34863_16
; NORMAL:  pushq  ${{-1|65535}}
; NORMAL-NEXT:  pushq  $0
; NORMAL-NEXT:  call
define void @pr34863_16(i16 %x) minsize nounwind {
entry:
  tail call void @eightparams16(i16 %x, i16 %x, i16 %x, i16 %x, i16 %x, i16 %x, i16 0, i16 -1)
  ret void
}

; NORMAL-LABEL: pr34863_32
; NORMAL:  pushq  ${{-1|65535}}
; NORMAL-NEXT:  pushq  $0
; NORMAL-NEXT:  call
define void @pr34863_32(i32 %x) minsize nounwind {
entry:
  tail call void @eightparams(i32 %x, i32 %x, i32 %x, i32 %x, i32 %x, i32 %x, i32 0, i32 -1)
  ret void
}

; NORMAL-LABEL: pr34863_64
; NORMAL:  pushq  ${{-1|65535}}
; NORMAL-NEXT:  pushq  $0
; NORMAL-NEXT:  call
define void @pr34863_64(i64 %x) minsize nounwind {
entry:
  tail call void @eightparams64(i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 0, i64 -1)
  ret void
}
