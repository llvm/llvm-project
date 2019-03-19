; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-apple-darwin -disable-block-placement | FileCheck --check-prefix=CHECK-APPLE %s
; RUN: llc -verify-machineinstrs -O0 < %s -mtriple=x86_64-apple-darwin -disable-block-placement | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -verify-machineinstrs < %s -mtriple=i386-apple-darwin -disable-block-placement | FileCheck --check-prefix=CHECK-i386 %s

declare i8* @malloc(i64)
declare void @free(i8*)
%swift_error = type {i64, i8}

; This tests the basic usage of a swifterror parameter. "foo" is the function
; that takes a swifterror parameter and "caller" is the caller of "foo".
define float @foo(%swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: foo:
; CHECK-APPLE: movl $16, %edi
; CHECK-APPLE: malloc
; CHECK-APPLE: movb $1, 8(%rax)
; CHECK-APPLE: movq %rax, %r12

; CHECK-O0-LABEL: foo:
; CHECK-O0: movl $16
; CHECK-O0: malloc
; CHECK-O0: movb $1, 8(%rax)
; CHECK-O0: movq %{{.*}}, %r12
entry:
  %call = call i8* @malloc(i64 16)
  %call.0 = bitcast i8* %call to %swift_error*
  store %swift_error* %call.0, %swift_error** %error_ptr_ref
  %tmp = getelementptr inbounds i8, i8* %call, i64 8
  store i8 1, i8* %tmp
  ret float 1.0
}

; "caller" calls "foo" that takes a swifterror parameter.
define float @caller(i8* %error_ref) {
; CHECK-APPLE-LABEL: caller:
; CHECK-APPLE: xorl %r12d, %r12d
; CHECK-APPLE: callq {{.*}}foo
; CHECK-APPLE: testq %r12, %r12
; CHECK-APPLE: jne
; Access part of the error object and save it to error_ref
; CHECK-APPLE: movb 8(%rdi)
; CHECK-APPLE: callq {{.*}}free

; CHECK-O0-LABEL: caller:
; CHECK-O0: xorl
; CHECK-O0: movl %{{.*}}, %r12d
; CHECK-O0: callq {{.*}}foo
; CHECK-O0: jne
entry:
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  %call = call float @foo(%swift_error** swifterror %error_ptr_ref)
  %error_from_foo = load %swift_error*, %swift_error** %error_ptr_ref
  %had_error_from_foo = icmp ne %swift_error* %error_from_foo, null
  %tmp = bitcast %swift_error* %error_from_foo to i8*
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %v1 = getelementptr inbounds %swift_error, %swift_error* %error_from_foo, i64 0, i32 1
  %t = load i8, i8* %v1
  store i8 %t, i8* %error_ref
  br label %handler
handler:
  call void @free(i8* %tmp)
  ret float 1.0
}

; "caller2" is the caller of "foo", it calls "foo" inside a loop.
define float @caller2(i8* %error_ref) {
; CHECK-APPLE-LABEL: caller2:
; CHECK-APPLE: xorl %r12d, %r12d
; CHECK-APPLE: callq {{.*}}foo
; CHECK-APPLE: testq %r12, %r12
; CHECK-APPLE: jne
; CHECK-APPLE: ucomiss
; CHECK-APPLE: jbe
; Access part of the error object and save it to error_ref
; CHECK-APPLE: movb 8(%r12)
; CHECK-APPLE: movq %r12, %rdi
; CHECK-APPLE: callq {{.*}}free

; CHECK-O0-LABEL: caller2:
; CHECK-O0: xorl
; CHECK-O0: movl %{{.*}}, %r12d
; CHECK-O0: callq {{.*}}foo
; CHECK-O0: movq %r12, [[ID:%[a-z]+]]
; CHECK-O0: cmpq $0, %r12
; CHECK-O0: jne
entry:
  %error_ptr_ref = alloca swifterror %swift_error*
  br label %bb_loop
bb_loop:
  store %swift_error* null, %swift_error** %error_ptr_ref
  %call = call float @foo(%swift_error** swifterror %error_ptr_ref)
  %error_from_foo = load %swift_error*, %swift_error** %error_ptr_ref
  %had_error_from_foo = icmp ne %swift_error* %error_from_foo, null
  %tmp = bitcast %swift_error* %error_from_foo to i8*
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %cmp = fcmp ogt float %call, 1.000000e+00
  br i1 %cmp, label %bb_end, label %bb_loop
bb_end:
  %v1 = getelementptr inbounds %swift_error, %swift_error* %error_from_foo, i64 0, i32 1
  %t = load i8, i8* %v1
  store i8 %t, i8* %error_ref
  br label %handler
handler:
  call void @free(i8* %tmp)
  ret float 1.0
}

; "foo_if" is a function that takes a swifterror parameter, it sets swifterror
; under a certain condition.
define float @foo_if(%swift_error** swifterror %error_ptr_ref, i32 %cc) {
; CHECK-APPLE-LABEL: foo_if:
; CHECK-APPLE: testl %edi, %edi
; CHECK-APPLE: je
; CHECK-APPLE: movl $16, %edi
; CHECK-APPLE: malloc
; CHECK-APPLE: movb $1, 8(%rax)
; CHECK-APPLE: movq %rax, %r12
; CHECK-APPLE-NOT: %r12
; CHECK-APPLE: ret

; CHECK-O0-LABEL: foo_if:
; CHECK-O0: cmpl $0
; spill to stack
; CHECK-O0: movq %r12, {{.*}}(%rsp)
; CHECK-O0: je
; CHECK-O0: movl $16,
; CHECK-O0: malloc
; CHECK-O0: movq %rax, [[ID:%[a-z]+]]
; CHECK-O0-DAG: movb $1, 8(%rax)
; CHECK-O0-DAG: movq [[ID]], %r12
; CHECK-O0: ret
; reload from stack
; CHECK-O0: movq {{.*}}(%rsp), [[REG:%[a-z]+]]
; CHECK-O0: movq [[REG]], %r12
; CHECK-O0: ret
entry:
  %cond = icmp ne i32 %cc, 0
  br i1 %cond, label %gen_error, label %normal

gen_error:
  %call = call i8* @malloc(i64 16)
  %call.0 = bitcast i8* %call to %swift_error*
  store %swift_error* %call.0, %swift_error** %error_ptr_ref
  %tmp = getelementptr inbounds i8, i8* %call, i64 8
  store i8 1, i8* %tmp
  ret float 1.0

normal:
  ret float 0.0
}

; "foo_loop" is a function that takes a swifterror parameter, it sets swifterror
; under a certain condition inside a loop.
define float @foo_loop(%swift_error** swifterror %error_ptr_ref, i32 %cc, float %cc2) {
; CHECK-APPLE-LABEL: foo_loop:
; CHECK-APPLE: movq %r12, %rax
; CHECK-APPLE: testl
; CHECK-APPLE: je
; CHECK-APPLE: movl $16, %edi
; CHECK-APPLE: malloc
; CHECK-APPLE: movb $1, 8(%rax)
; CHECK-APPLE: ucomiss
; CHECK-APPLE: jbe
; CHECK-APPLE: movq %rax, %r12
; CHECK-APPLE: ret

; CHECK-O0-LABEL: foo_loop:
; spill to stack
; CHECK-O0: movq %r12, {{.*}}(%rsp)
; CHECK-O0: cmpl $0
; CHECK-O0: je
; CHECK-O0: movl $16,
; CHECK-O0: malloc
; CHECK-O0: movq %rax, [[ID:%[a-z0-9]+]]
; CHECK-O0: movb $1, 8([[ID]])
; CHECK-O0: jbe
; reload from stack
; CHECK-O0: movq {{.*}}(%rsp), [[REG:%[a-z0-9]+]]
; CHECK-O0: movq [[REG]], %r12
; CHECK-O0: ret
entry:
  br label %bb_loop

bb_loop:
  %cond = icmp ne i32 %cc, 0
  br i1 %cond, label %gen_error, label %bb_cont

gen_error:
  %call = call i8* @malloc(i64 16)
  %call.0 = bitcast i8* %call to %swift_error*
  store %swift_error* %call.0, %swift_error** %error_ptr_ref
  %tmp = getelementptr inbounds i8, i8* %call, i64 8
  store i8 1, i8* %tmp
  br label %bb_cont

bb_cont:
  %cmp = fcmp ogt float %cc2, 1.000000e+00
  br i1 %cmp, label %bb_end, label %bb_loop
bb_end:
  ret float 0.0
}

%struct.S = type { i32, i32, i32, i32, i32, i32 }

; "foo_sret" is a function that takes a swifterror parameter, it also has a sret
; parameter.
define void @foo_sret(%struct.S* sret %agg.result, i32 %val1, %swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: foo_sret:
; CHECK-APPLE: movq %rdi, %{{.*}}
; CHECK-APPLE: movl $16, %edi
; CHECK-APPLE: malloc
; CHECK-APPLE: movb $1, 8(%rax)
; CHECK-APPLE: movl %{{.*}}, 4(%{{.*}})
; CHECK-APPLE: movq %rax, %r12
; CHECK-APPLE: movq %{{.*}}, %rax
; CHECK-APPLE-NOT: x19

; CHECK-O0-LABEL: foo_sret:
; CHECK-O0: movl $16,
; spill sret to stack
; CHECK-O0: movq %rdi,
; CHECK-O0: movq {{.*}}, %rdi
; CHECK-O0: malloc
; CHECK-O0: movb $1, 8(%rax)
; CHECK-O0: movl %{{.*}}, 4(%{{.*}})
; CHECK-O0: movq %{{.*}}, %r12
; reload sret from stack
; CHECK-O0: movq {{.*}}(%rsp), %rax
; CHECK-O0: ret
entry:
  %call = call i8* @malloc(i64 16)
  %call.0 = bitcast i8* %call to %swift_error*
  store %swift_error* %call.0, %swift_error** %error_ptr_ref
  %tmp = getelementptr inbounds i8, i8* %call, i64 8
  store i8 1, i8* %tmp
  %v2 = getelementptr inbounds %struct.S, %struct.S* %agg.result, i32 0, i32 1
  store i32 %val1, i32* %v2
  ret void
}

; "caller3" calls "foo_sret" that takes a swifterror parameter.
define float @caller3(i8* %error_ref) {
; CHECK-APPLE-LABEL: caller3:
; CHECK-APPLE: movl $1, %esi
; CHECK-APPLE: xorl %r12d, %r12d
; CHECK-APPLE: callq {{.*}}foo_sret
; CHECK-APPLE: testq %r12, %r12
; CHECK-APPLE: jne
; Access part of the error object and save it to error_ref
; CHECK-APPLE: movb 8(%rdi),
; CHECK-APPLE: movb %{{.*}},
; CHECK-APPLE: callq {{.*}}free

; CHECK-O0-LABEL: caller3:
; CHECK-O0: xorl
; CHECK-O0: movl {{.*}}, %r12d
; CHECK-O0: movl $1, %esi
; CHECK-O0: movq {{.*}}, %rdi
; CHECK-O0: callq {{.*}}foo_sret
; CHECK-O0: movq %r12,
; CHECK-O0: cmpq $0
; CHECK-O0: jne
; Access part of the error object and save it to error_ref
; CHECK-O0: movb 8(%{{.*}}),
; CHECK-O0: movb %{{.*}},
; reload from stack
; CHECK-O0: movq {{.*}}(%rsp), %rdi
; CHECK-O0: callq {{.*}}free
entry:
  %s = alloca %struct.S, align 8
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  call void @foo_sret(%struct.S* sret %s, i32 1, %swift_error** swifterror %error_ptr_ref)
  %error_from_foo = load %swift_error*, %swift_error** %error_ptr_ref
  %had_error_from_foo = icmp ne %swift_error* %error_from_foo, null
  %tmp = bitcast %swift_error* %error_from_foo to i8*
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %v1 = getelementptr inbounds %swift_error, %swift_error* %error_from_foo, i64 0, i32 1
  %t = load i8, i8* %v1
  store i8 %t, i8* %error_ref
  br label %handler
handler:
  call void @free(i8* %tmp)
  ret float 1.0
}

; This is a caller with multiple swifterror values, it calls "foo" twice, each
; time with a different swifterror value, from "alloca swifterror".
define float @caller_with_multiple_swifterror_values(i8* %error_ref, i8* %error_ref2) {
; CHECK-APPLE-LABEL: caller_with_multiple_swifterror_values:

; The first swifterror value:
; CHECK-APPLE: xorl %r12d, %r12d
; CHECK-APPLE: callq {{.*}}foo
; CHECK-APPLE: testq %r12, %r12
; CHECK-APPLE: jne
; Access part of the error object and save it to error_ref
; CHECK-APPLE: movb 8(%rdi)
; CHECK-APPLE: callq {{.*}}free

; The second swifterror value:
; CHECK-APPLE: xorl %r12d, %r12d
; CHECK-APPLE: callq {{.*}}foo
; CHECK-APPLE: testq %r12, %r12
; CHECK-APPLE: jne
; Access part of the error object and save it to error_ref
; CHECK-APPLE: movb 8(%rdi)
; CHECK-APPLE: callq {{.*}}free

; CHECK-O0-LABEL: caller_with_multiple_swifterror_values:

; The first swifterror value:
; CHECK-O0: xorl
; CHECK-O0: movl %{{.*}}, %r12d
; CHECK-O0: callq {{.*}}foo
; CHECK-O0: jne

; The second swifterror value:
; CHECK-O0: xorl
; CHECK-O0: movl %{{.*}}, %r12d
; CHECK-O0: callq {{.*}}foo
; CHECK-O0: jne
entry:
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  %call = call float @foo(%swift_error** swifterror %error_ptr_ref)
  %error_from_foo = load %swift_error*, %swift_error** %error_ptr_ref
  %had_error_from_foo = icmp ne %swift_error* %error_from_foo, null
  %tmp = bitcast %swift_error* %error_from_foo to i8*
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %v1 = getelementptr inbounds %swift_error, %swift_error* %error_from_foo, i64 0, i32 1
  %t = load i8, i8* %v1
  store i8 %t, i8* %error_ref
  br label %handler
handler:
  call void @free(i8* %tmp)

  %error_ptr_ref2 = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref2
  %call2 = call float @foo(%swift_error** swifterror %error_ptr_ref2)
  %error_from_foo2 = load %swift_error*, %swift_error** %error_ptr_ref2
  %had_error_from_foo2 = icmp ne %swift_error* %error_from_foo2, null
  %bitcast2 = bitcast %swift_error* %error_from_foo2 to i8*
  br i1 %had_error_from_foo2, label %handler2, label %cont2
cont2:
  %v2 = getelementptr inbounds %swift_error, %swift_error* %error_from_foo2, i64 0, i32 1
  %t2 = load i8, i8* %v2
  store i8 %t2, i8* %error_ref2
  br label %handler2
handler2:
  call void @free(i8* %bitcast2)

  ret float 1.0
}

%swift.refcounted = type opaque

; This test checks that we don't create bad phi nodes as part of swifterror
; isel. We used to fail machine ir verification.
; CHECK-APPLE: _swifterror_isel
; CHECK-O0: _swifterror_isel
define void @swifterror_isel(%swift.refcounted*) {
entry:
  %swifterror = alloca swifterror %swift_error*, align 8
  br i1 undef, label %5, label %1

  %2 = phi i16 [ %4, %1 ], [ undef, %entry ]
  %3 = call i1 undef(i16 %2, %swift.refcounted* swiftself %0, %swift_error** nocapture swifterror %swifterror)
  %4 = load i16, i16* undef, align 2
  br label %1

  ret void
}

; This tests the basic usage of a swifterror parameter with swiftcc.
define swiftcc float @foo_swiftcc(%swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: foo_swiftcc:
; CHECK-APPLE: movl $16, %edi
; CHECK-APPLE: malloc
; CHECK-APPLE: movb $1, 8(%rax)
; CHECK-APPLE: movq %rax, %r12

; CHECK-O0-LABEL: foo_swiftcc:
; CHECK-O0: movl $16
; CHECK-O0: malloc
; CHECK-O0: movb $1, 8(%rax)
; CHECK-O0: movq %{{.*}}, %r12
entry:
  %call = call i8* @malloc(i64 16)
  %call.0 = bitcast i8* %call to %swift_error*
  store %swift_error* %call.0, %swift_error** %error_ptr_ref
  %tmp = getelementptr inbounds i8, i8* %call, i64 8
  store i8 1, i8* %tmp
  ret float 1.0
}

declare swiftcc float @moo(%swift_error** swifterror)

; Test parameter forwarding.
define swiftcc float @forward_swifterror(%swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: forward_swifterror:
; CHECK-APPLE: pushq %rax
; CHECK-APPLE: callq _moo
; CHECK-APPLE: popq %rax
; CHECK-APPLE: retq

; CHECK-O0-LABEL: forward_swifterror:
; CHECK-O0: pushq %rax
; CHECK-O0:  callq _moo
; CHECK-O0: popq %rax
; CHECK-O0:  retq

entry:
  %call = call swiftcc float @moo(%swift_error** swifterror %error_ptr_ref)
  ret float %call
}

define swiftcc float @conditionally_forward_swifterror(%swift_error** swifterror %error_ptr_ref, i32 %cc) {
; CHECK-APPLE-LABEL: conditionally_forward_swifterror:
; CHECK-APPLE:  pushq %rax
; CHECK-APPLE:	testl %edi, %edi
; CHECK-APPLE:  je

; CHECK-APPLE:  callq _moo
; CHECK-APPLE:  popq %rax
; CHECK-APPLE:  retq

; CHECK-APPLE:  xorps %xmm0, %xmm0
; CHECK-APPLE:  popq %rax
; CHECK-APPLE:  retq

; CHECK-O0-LABEL: conditionally_forward_swifterror:
; CHECK-O0: pushq [[REG1:%[a-z0-9]+]]
; CHECK-O0:  cmpl $0, %edi
; CHECK-O0-DAG:  movq %r12, (%rsp)
; CHECK-O0:  je

; CHECK-O0:  movq (%rsp), [[REG:%[a-z0-9]+]]
; CHECK-O0:  movq [[REG]], %r12
; CHECK-O0:  callq _moo
; CHECK-O0:  popq [[REG1]]
; CHECK-O0:  retq

; CHECK-O0:  movq (%rsp), [[REG:%[a-z0-9]+]]
; CHECK-O0:  xorps %xmm0, %xmm0
; CHECK-O0:  movq [[REG]], %r12
; CHECK-O0:  popq [[REG1]]
; CHECK-O0:  retq
entry:
  %cond = icmp ne i32 %cc, 0
  br i1 %cond, label %gen_error, label %normal

gen_error:
  %call = call swiftcc float @moo(%swift_error** swifterror %error_ptr_ref)
  ret float %call

normal:
  ret float 0.0
}

; Check that we don't blow up on tail calling swifterror argument functions.
define float @tailcallswifterror(%swift_error** swifterror %error_ptr_ref) {
entry:
  %0 = tail call float @tailcallswifterror(%swift_error** swifterror %error_ptr_ref)
  ret float %0
}
define swiftcc float @tailcallswifterror_swiftcc(%swift_error** swifterror %error_ptr_ref) {
entry:
  %0 = tail call swiftcc float @tailcallswifterror_swiftcc(%swift_error** swifterror %error_ptr_ref)
  ret float %0
}

; Check that we can handle an empty function with swifterror argument.
; CHECK-i386-LABEL: empty_swiftcc:
; CHECK-i386:  movl    4(%esp), %eax
; CHECK-i386:  movl    8(%esp), %edx
; CHECK-i386:  movl    12(%esp), %ecx
; CHECK-i386:  retl
; CHECK-APPLE-LABEL: empty_swiftcc:
; CHECK-APPLE:  movl    %edx, %ecx
; CHECK-APPLE-DAG:  movl    %edi, %eax
; CHECK-APPLE-DAG:  movl    %esi, %edx
; CHECK-APPLE:  retq
define swiftcc {i32, i32, i32} @empty_swiftcc({i32, i32, i32} , %swift_error** swifterror %error_ptr_ref) {
entry:
  ret {i32, i32, i32} %0
}

; Make sure we can handle the case when isel generates new machine basic blocks.
; CHECK-APPLE-LABEL: dont_crash_on_new_isel_blocks:
; CHECK-APPLE: pushq   %rax
; CHECK-APPLE: xorl    %eax, %eax
; CHECK-APPLE: testb   %al, %al
; CHECK-APPLE: jne
; CHECK-APPLE: callq   *%rax
; CHECK-APPLE: popq    %rax
; CHECK-APPLE: ret

define swiftcc void @dont_crash_on_new_isel_blocks(%swift_error** nocapture swifterror, i1, i8**) {
entry:
  %3 = or i1 false, %1
  br i1 %3, label %cont, label %falsebb

falsebb:
  %4 = load i8*, i8** %2, align 8
  br label %cont

cont:
  tail call swiftcc void undef(%swift_error** nocapture swifterror %0)
  ret void
}

; CHECK-APPLE-LABEL: swifterror_clobber
; CHECK-APPLE: movq %r12, [[REG:%.*]]
; CHECK-APPLE: nop
; CHECK-APPLE: movq [[REG]], %r12
define swiftcc void @swifterror_clobber(%swift_error** nocapture swifterror %err) {
  call void asm sideeffect "nop", "~{r12}"()
  ret void
}

; CHECK-APPLE-LABEL: swifterror_reg_clobber
; CHECK-APPLE: pushq %r12
; CHECK-APPLE: nop
; CHECK-APPLE: popq  %r12
define swiftcc void @swifterror_reg_clobber(%swift_error** nocapture %err) {
  call void asm sideeffect "nop", "~{r12}"()
  ret void
}

; CHECK-APPLE-LABEL: params_in_reg
; Save callee save registers to store clobbered arguments.
; CHECK-APPLE:  pushq   %rbp
; CHECK-APPLE:  pushq   %r15
; CHECK-APPLE:  pushq   %r14
; Clobbered swiftself register.
; CHECK-APPLE:  pushq   %r13
; CHECK-APPLE:  pushq   %rbx
; CHECK-APPLE:  subq    $48, %rsp
; Save arguments.
; CHECK-APPLE:  movq    %r12, 32(%rsp)
; CHECK-APPLE:  movq    %r13, 24(%rsp)
; CHECK-APPLE:  movq    %r9, 16(%rsp)
; CHECK-APPLE:  movq    %r8, 8(%rsp)
; CHECK-APPLE:  movq    %rcx, %r14
; CHECK-APPLE:  movq    %rdx, %r15
; CHECK-APPLE:  movq    %rsi, %rbx
; CHECK-APPLE:  movq    %rdi, %rbp
; Setup call.
; CHECK-APPLE:  movl    $1, %edi
; CHECK-APPLE:  movl    $2, %esi
; CHECK-APPLE:  movl    $3, %edx
; CHECK-APPLE:  movl    $4, %ecx
; CHECK-APPLE:  movl    $5, %r8d
; CHECK-APPLE:  movl    $6, %r9d
; CHECK-APPLE:  xorl    %r13d, %r13d
; CHECK-APPLE:  xorl    %r12d, %r12d
; CHECK-APPLE:  callq   _params_in_reg2
; Setup second call with stored arguments.
; CHECK-APPLE:  movq    %rbp, %rdi
; CHECK-APPLE:  movq    %rbx, %rsi
; CHECK-APPLE:  movq    %r15, %rdx
; CHECK-APPLE:  movq    %r14, %rcx
; CHECK-APPLE:  movq    8(%rsp), %r8
; CHECK-APPLE:  movq    16(%rsp), %r9
; CHECK-APPLE:  movq    24(%rsp), %r13
; CHECK-APPLE:  movq    32(%rsp), %r12
; CHECK-APPLE:  callq   _params_in_reg2
; CHECK-APPLE:  addq    $48, %rsp
; CHECK-APPLE:  popq    %rbx
; CHECK-APPLE:  popq    %r13
; CHECK-APPLE:  popq    %r14
; CHECK-APPLE:  popq    %r15
; CHECK-APPLE:  popq    %rbp
define swiftcc void @params_in_reg(i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  call swiftcc void @params_in_reg2(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i8* swiftself %6, %swift_error** nocapture swifterror %err)
  ret void
}
declare swiftcc void @params_in_reg2(i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err)

; CHECK-APPLE-LABEL: params_and_return_in_reg
; CHECK-APPLE:  pushq   %rbp
; CHECK-APPLE:  pushq   %r15
; CHECK-APPLE:  pushq   %r14
; CHECK-APPLE:  pushq   %r13
; CHECK-APPLE:  pushq   %rbx
; CHECK-APPLE:  subq    $48, %rsp
; Store arguments.
; CHECK-APPLE:  movq    %r12, %r14
; CHECK-APPLE:  movq    %r13, (%rsp)
; CHECK-APPLE:  movq    %r9, 32(%rsp)
; CHECK-APPLE:  movq    %r8, 24(%rsp)
; CHECK-APPLE:  movq    %rcx, 16(%rsp)
; CHECK-APPLE:  movq    %rdx, %r15
; CHECK-APPLE:  movq    %rsi, %rbx
; CHECK-APPLE:  movq    %rdi, %rbp
; Setup call that clobbers all argument registers.
; CHECK-APPLE:  movl    $1, %edi
; CHECK-APPLE:  movl    $2, %esi
; CHECK-APPLE:  movl    $3, %edx
; CHECK-APPLE:  movl    $4, %ecx
; CHECK-APPLE:  movl    $5, %r8d
; CHECK-APPLE:  movl    $6, %r9d
; CHECK-APPLE:  xorl    %r13d, %r13d
; CHECK-APPLE:  xorl    %r12d, %r12d
; CHECK-APPLE:  callq   _params_in_reg2
; Store error_ptr_ref for later use.
; CHECK-APPLE:  movq    %r12, 8(%rsp)
; Restore original arguments.
; CHECK-APPLE:  movq    %rbp, %rdi
; CHECK-APPLE:  movq    %rbx, %rsi
; CHECK-APPLE:  movq    %r15, %rdx
; CHECK-APPLE:  movq    16(%rsp), %rcx
; CHECK-APPLE:  movq    24(%rsp), %r8
; CHECK-APPLE:  movq    32(%rsp), %r9
; CHECK-APPLE:  movq    (%rsp), %r13
; CHECK-APPLE:  movq    %r14, %r12
; CHECK-APPLE:  callq   _params_and_return_in_reg2
; Store return values in callee saved registers.
; CHECK-APPLE:  movq    %rax, %rbx
; CHECK-APPLE:  movq    %rdx, %rbp
; CHECK-APPLE:  movq    %rcx, %r15
; CHECK-APPLE:  movq    %r8, %r14
; Store the swifterror return value (%err).
; CHECK-APPLE:  movq    %r12, (%rsp)
; Setup call.
; CHECK-APPLE:  movl    $1, %edi
; CHECK-APPLE:  movl    $2, %esi
; CHECK-APPLE:  movl    $3, %edx
; CHECK-APPLE:  movl    $4, %ecx
; CHECK-APPLE:  movl    $5, %r8d
; CHECK-APPLE:  movl    $6, %r9d
; CHECK-APPLE:  xorl    %r13d, %r13d
; Restore the swifterror value of error_ptr_ref.
; CHECK-APPLE:  movq    8(%rsp), %r12
; CHECK-APPLE:  callq   _params_in_reg2
; Restore the return values of _params_and_return_in_reg2.
; CHECK-APPLE:  movq    %rbx, %rax
; CHECK-APPLE:  movq    %rbp, %rdx
; CHECK-APPLE:  movq    %r15, %rcx
; CHECK-APPLE:  movq    %r14, %r8
; Restore the swiferror value of err.
; CHECK-APPLE:  movq    (%rsp), %r12
; CHECK-APPLE:  addq    $48, %rsp
; CHECK-APPLE:  popq    %rbx
; CHECK-APPLE:  popq    %r13
; CHECK-APPLE:  popq    %r14
; CHECK-APPLE:  popq    %r15
; CHECK-APPLE:  popq    %rbp
define swiftcc { i64, i64, i64, i64} @params_and_return_in_reg(i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  %val = call swiftcc  { i64, i64, i64, i64 } @params_and_return_in_reg2(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i8* swiftself %6, %swift_error** nocapture swifterror %err)
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  ret { i64, i64, i64, i64 }%val
}

declare swiftcc { i64, i64, i64, i64 } @params_and_return_in_reg2(i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err)


declare void @acallee(i8*)

; Make sure we don't tail call if the caller returns a swifterror value. We
; would have to move into the swifterror register before the tail call.
; CHECK-APPLE: tailcall_from_swifterror:
; CHECK-APPLE-NOT: jmp _acallee
; CHECK-APPLE: callq _acallee

define swiftcc void @tailcall_from_swifterror(%swift_error** swifterror %error_ptr_ref) {
entry:
  tail call void @acallee(i8* null)
  ret void
}

; Make sure we don't crash on this function during -O0.
; We used to crash because we would insert an IMPLICIT_DEF for the swifterror at
; beginning of the machine basic block but did not inform FastISel of the
; inserted instruction. When computing the InsertPoint in the entry block
; FastISel would choose an insertion point before the IMPLICIT_DEF causing a
; crash later on.
declare hidden swiftcc i8* @testFunA()

%TSb = type <{ i1 }>

define swiftcc void @dontCrash()  {
entry:
  %swifterror = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %swifterror, align 8
  %a = call i8* @testFunA()
  %b = bitcast i8* %a to %TSb*
  %._value = getelementptr inbounds %TSb, %TSb* %b, i32 0, i32 0
  %c = load i1, i1* %._value, align 1
  br i1 %c, label %trueBB, label %falseBB

trueBB:
  ret void

falseBB:
  ret void
}


declare swiftcc void @foo2(%swift_error** swifterror)

; Make sure we properly assign registers during fast-isel.
; CHECK-O0-LABEL: testAssign
; CHECK-O0:        pushq   %r12
; CHECK-O0:        xorl    [[ZERO:%[a-z0-9]+]], [[ZERO]]
; CHECK-O0:        movl    [[ZERO]], %r12d
; CHECK-O0:        callq   _foo2
; CHECK-O0:        movq    %r12, [[SLOT:[-a-z0-9\(\)\%]*]]
;
; CHECK-O0:        movq    [[SLOT]], %rax
; CHECK-O0:        popq    %r12
; CHECK-O0:        retq

; CHECK-APPLE-LABEL: testAssign
; CHECK-APPLE:        pushq   %r12
; CHECK-APPLE:        xorl    %r12d, %r12d
; CHECK-APPLE:        callq   _foo2
; CHECK-APPLE:        movq    %r12, %rax
; CHECK-APPLE:        popq    %r12
; CHECK-APPLE:        retq

define swiftcc %swift_error* @testAssign(i8* %error_ref) {
entry:
  %error_ptr = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr
  call swiftcc void @foo2(%swift_error** swifterror %error_ptr)
  br label %a

a:
  %error = load %swift_error*, %swift_error** %error_ptr
  ret %swift_error* %error
}

; CHECK-O0-LABEL: testAssign2
; CHECK-O0:        movq    %r12, [[SLOT:[-a-z0-9\(\)\%]*]]
; CHECK-O0:        jmp
; CHECK-O0:        movq    [[SLOT]], %rax
; CHECK-O0:        movq    %rax, [[SLOT2:[-a-z0-9\(\)\%]*]]
; CHECK-O0:        movq    [[SLOT2]], %r12
; CHECK-O0:        retq

; CHECK-APPLE-LABEL: testAssign2
; CHECK-APPLE:        movq    %r12, %rax
; CHECK-APPLE:        retq
define swiftcc %swift_error* @testAssign2(i8* %error_ref, %swift_error** swifterror %err) {
entry:
  br label %a

a:
  %error = load %swift_error*, %swift_error** %err
  ret %swift_error* %error
}

; CHECK-O0-LABEL: testAssign3
; CHECK-O0:        callq   _foo2
; CHECK-O0:        movq    %r12, [[SLOT:[-a-z0-9\(\)\%]*]]
; CHECK-O0:        movq    [[SLOT]], %rax
; CHECK-O0:        movq    %rax, [[SLOT2:[-a-z0-9\(\)\%]*]]
; CHECK-O0:        movq    [[SLOT2]], %r12
; CHECK-O0:        addq    $24, %rsp
; CHECK-O0:        retq

; CHECK-APPLE-LABEL: testAssign3
; CHECK-APPLE:         callq   _foo2
; CHECK-APPLE:         movq    %r12, %rax
; CHECK-APPLE:         retq

define swiftcc %swift_error* @testAssign3(i8* %error_ref, %swift_error** swifterror %err) {
entry:
  call swiftcc void @foo2(%swift_error** swifterror %err)
  br label %a

a:
  %error = load %swift_error*, %swift_error** %err
  ret %swift_error* %error
}


; CHECK-O0-LABEL: testAssign4
; CHECK-O0:        callq   _foo2
; CHECK-O0:        xorl    %eax, %eax
; CHECK-O0: ## kill: def $rax killed $eax
; CHECK-O0:        movq    %rax, [[SLOT:[-a-z0-9\(\)\%]*]]
; CHECK-O0:        movq    [[SLOT]], %rax
; CHECK-O0:        movq    %rax, [[SLOT2:[-a-z0-9\(\)\%]*]]
; CHECK-O0:        movq    [[SLOT2]], %r12
; CHECK-O0:        retq

; CHECK-APPLE-LABEL: testAssign4
; CHECK-APPLE:        callq   _foo2
; CHECK-APPLE:        xorl    %eax, %eax
; CHECK-APPLE:        xorl    %r12d, %r12d
; CHECK-APPLE:        retq

define swiftcc %swift_error* @testAssign4(i8* %error_ref, %swift_error** swifterror %err) {
entry:
  call swiftcc void @foo2(%swift_error** swifterror %err)
  store %swift_error* null, %swift_error** %err
  br label %a

a:
  %error = load %swift_error*, %swift_error** %err
  ret %swift_error* %error
}
