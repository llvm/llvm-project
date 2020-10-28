; RUN: llc -verify-machineinstrs -frame-pointer=all -global-isel < %s -mtriple=aarch64-apple-ios -disable-post-ra | FileCheck %s

declare i8* @malloc(i64)
declare void @free(i8*)
%swift_error = type {i64, i8}

; This tests the basic usage of a swifterror parameter. "foo" is the function
; that takes a swifterror parameter and "caller" is the caller of "foo".
define float @foo(%swift_error** swifterror %error_ptr_ref) {
; CHECK-LABEL: foo:
; CHECK: mov w0, #16
; CHECK: malloc
; CHECK: mov [[ID:w[0-9]+]], #1
; CHECK: strb [[ID]], [x0, #8]
; CHECK: mov x21, x0
; CHECK-NOT: x21

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
; CHECK-LABEL: caller:
; CHECK: mov [[ID:x[0-9]+]], x0
; CHECK: bl {{.*}}foo
; CHECK: mov x0, x21
; CHECK: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK: bl {{.*}}free

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
; CHECK-LABEL: caller2:
; CHECK: mov [[ID:x[0-9]+]], x0
; CHECK: fmov [[CMP:s[0-9]+]], #1.0
; CHECK: mov x21, xzr
; CHECK: bl {{.*}}foo
; CHECK: cbnz x21
; CHECK: fcmp s0, [[CMP]]
; CHECK: b.le
; Access part of the error object and save it to error_ref
; CHECK: ldrb [[CODE:w[0-9]+]], [x21, #8]
; CHECK: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK: mov x0, x21
; CHECK: bl {{.*}}free

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
; CHECK-LABEL: foo_if:
; CHECK: cbz w0
; CHECK: mov w0, #16
; CHECK: malloc
; CHECK: mov [[ID:w[0-9]+]], #1
; CHECK: strb [[ID]], [x0, #8]
; CHECK: mov x21, x0
; CHECK-NOT: x21
; CHECK: ret

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
; CHECK-LABEL: foo_loop:
; CHECK: cbz
; CHECK: mov w0, #16
; CHECK: malloc
; CHECK: mov x21, x0
; CHECK: strb w{{.*}}, [x0, #8]
; CHECK: ret

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
; CHECK-LABEL: foo_sret:
; CHECK: mov [[SRET:x[0-9]+]], x8
; CHECK: mov w0, #16
; CHECK: malloc
; CHECK: mov [[ID:w[0-9]+]], #1
; CHECK: strb [[ID]], [x0, #8]
; CHECK: str w{{.*}}, [{{.*}}[[SRET]], #4]
; CHECK: mov x21, x0
; CHECK-NOT: x21

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
; CHECK-LABEL: caller3:
; CHECK: mov [[ID:x[0-9]+]], x0
; CHECK: mov [[ZERO:x[0-9]+]], xzr
; CHECK: bl {{.*}}foo_sret
; CHECK: mov x0, x21
; CHECK: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK: bl {{.*}}free

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

; "foo_vararg" is a function that takes a swifterror parameter, it also has
; variable number of arguments.
declare void @llvm.va_start(i8*) nounwind
define float @foo_vararg(%swift_error** swifterror %error_ptr_ref, ...) {
; CHECK-LABEL: foo_vararg:
; CHECK: mov w0, #16
; CHECK: malloc
; CHECK-DAG: mov [[ID:w[0-9]+]], #1
; CHECK-DAG: strb [[ID]], [x0, #8]

; First vararg
; CHECK: ldr {{w[0-9]+}}, [x[[ARG1:[0-9]+]]], #8
; Second vararg
; CHECK: ldr {{w[0-9]+}}, [x[[ARG1]]], #8
; Third vararg
; CHECK: ldr {{w[0-9]+}}, [x[[ARG1]]], #8

; CHECK: mov x21, x0
; CHECK-NOT: x21
entry:
  %call = call i8* @malloc(i64 16)
  %call.0 = bitcast i8* %call to %swift_error*
  store %swift_error* %call.0, %swift_error** %error_ptr_ref
  %tmp = getelementptr inbounds i8, i8* %call, i64 8
  store i8 1, i8* %tmp

  %args = alloca i8*, align 8
  %a10 = alloca i32, align 4
  %a11 = alloca i32, align 4
  %a12 = alloca i32, align 4
  %v10 = bitcast i8** %args to i8*
  call void @llvm.va_start(i8* %v10)
  %v11 = va_arg i8** %args, i32
  store i32 %v11, i32* %a10, align 4
  %v12 = va_arg i8** %args, i32
  store i32 %v12, i32* %a11, align 4
  %v13 = va_arg i8** %args, i32
  store i32 %v13, i32* %a12, align 4

  ret float 1.0
}

; "caller4" calls "foo_vararg" that takes a swifterror parameter.
define float @caller4(i8* %error_ref) {
; CHECK-LABEL: caller4:

; CHECK: mov [[ID:x[0-9]+]], x0
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; CHECK: str {{x[0-9]+}}, [sp, #16]
; CHECK: mov x21, xzr

; CHECK: bl {{.*}}foo_vararg
; CHECK: mov x0, x21
; CHECK: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK: bl {{.*}}free
entry:
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref

  %a10 = alloca i32, align 4
  %a11 = alloca i32, align 4
  %a12 = alloca i32, align 4
  store i32 10, i32* %a10, align 4
  store i32 11, i32* %a11, align 4
  store i32 12, i32* %a12, align 4
  %v10 = load i32, i32* %a10, align 4
  %v11 = load i32, i32* %a11, align 4
  %v12 = load i32, i32* %a12, align 4

  %call = call float (%swift_error**, ...) @foo_vararg(%swift_error** swifterror %error_ptr_ref, i32 %v10, i32 %v11, i32 %v12)
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

; CHECK-LABEL: params_in_reg
; Save callee saved registers and swifterror since it will be clobbered by the first call to params_in_reg2.
; CHECK:  stp     x28, x0, [sp
; CHECK:  stp     x27, x26, [sp
; CHECK:  stp     x25, x24, [sp
; CHECK:  stp     x23, x22, [sp
; CHECK:  stp     x20, x19, [sp
; CHECK:  stp     x29, x30, [sp
; Store argument registers.
; CHECK:  mov      x20, x1
; CHECK:  mov      x22, x2
; CHECK:  mov      x23, x3
; CHECK:  mov      x24, x4
; CHECK:  mov      x25, x5
; CHECK:  mov      x26, x6
; CHECK:  mov      x27, x7
; CHECK:  mov      x28, x21
; Setup call.
; CHECK:  mov     w0, #1
; CHECK:  mov     w1, #2
; CHECK:  mov     w2, #3
; CHECK:  mov     w3, #4
; CHECK:  mov     w4, #5
; CHECK:  mov     w5, #6
; CHECK:  mov     w6, #7
; CHECK:  mov     w7, #8
; CHECK:  str     xzr, [sp]
; CHECK:  mov      x21, xzr
; CHECK:  bl      _params_in_reg2
; Restore original arguments for next call.
; CHECK:  ldr      x0, [sp
; CHECK:  mov      x1, x20
; CHECK:  mov      x2, x22
; CHECK:  mov      x3, x23
; CHECK:  mov      x4, x24
; CHECK:  mov      x5, x25
; CHECK:  mov      x6, x26
; CHECK:  mov      x7, x27
; Restore original swiftself argument and swifterror %err.
; CHECK:  mov      x21, x28
; CHECK:  bl      _params_in_reg2
; Restore calle save registers but don't clober swifterror x21.
; CHECK-NOT: x21
; CHECK:  ldp     x29, x30, [sp
; CHECK-NOT: x21
; CHECK:  ldp     x20, x19, [sp
; CHECK-NOT: x21
; CHECK:  ldp     x23, x22, [sp
; CHECK-NOT: x21
; CHECK:  ldp     x25, x24, [sp
; CHECK-NOT: x21
; CHECK:  ldp     x27, x26, [sp
; CHECK-NOT: x21
; CHECK:  ldr     x28, [sp
; CHECK-NOT: x21
; CHECK:  ret
define swiftcc void @params_in_reg(i64, i64, i64, i64, i64, i64, i64, i64, i8*, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i8*  null, %swift_error** nocapture swifterror %error_ptr_ref)
  call swiftcc void @params_in_reg2(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i8*  %8, %swift_error** nocapture swifterror %err)
  ret void
}
declare swiftcc void @params_in_reg2(i64, i64, i64, i64, i64, i64, i64, i64, i8* , %swift_error** nocapture swifterror %err)

; CHECK-LABEL: params_and_return_in_reg
; Store callee saved registers.
; CHECK:  stp     x28, x0, [sp, #16
; CHECK:  stp     x27, x26, [sp
; CHECK:  stp     x25, x24, [sp
; CHECK:  stp     x23, x22, [sp
; CHECK:  stp     x20, x19, [sp
; CHECK:  stp     x29, x30, [sp
; Save original arguments.
; CHECK:  mov      x20, x1
; CHECK:  mov      x22, x2
; CHECK:  mov      x23, x3
; CHECK:  mov      x24, x4
; CHECK:  mov      x25, x5
; CHECK:  mov      x26, x6
; CHECK:  mov      x27, x7
; CHECK:  mov      x28, x21
; Setup call arguments.
; CHECK:  mov     w0, #1
; CHECK:  mov     w1, #2
; CHECK:  mov     w2, #3
; CHECK:  mov     w3, #4
; CHECK:  mov     w4, #5
; CHECK:  mov     w5, #6
; CHECK:  mov     w6, #7
; CHECK:  mov     w7, #8
; CHECK:  mov      x21, xzr
; CHECK:  bl      _params_in_reg2
; Store swifterror %error_ptr_ref.
; CHECK:  stp     {{x[0-9]+}}, x21, [sp]
; Setup call arguments from original arguments.
; CHECK:  ldr      x0, [sp, #24
; CHECK:  mov      x1, x20
; CHECK:  mov      x2, x22
; CHECK:  mov      x3, x23
; CHECK:  mov      x4, x24
; CHECK:  mov      x5, x25
; CHECK:  mov      x6, x26
; CHECK:  mov      x7, x27
; CHECK:  mov      x21, x28
; CHECK:  bl      _params_and_return_in_reg2
; Store return values.
; CHECK:  mov      x20, x0
; CHECK:  mov      x22, x1
; CHECK:  mov      x23, x2
; CHECK:  mov      x24, x3
; CHECK:  mov      x25, x4
; CHECK:  mov      x26, x5
; CHECK:  mov      x27, x6
; CHECK:  mov      x28, x7
; Save swifterror %err.
; CHECK:  mov      x19, x21
; Setup call.
; CHECK:  mov     w0, #1
; CHECK:  mov     w1, #2
; CHECK:  mov     w2, #3
; CHECK:  mov     w3, #4
; CHECK:  mov     w4, #5
; CHECK:  mov     w5, #6
; CHECK:  mov     w6, #7
; CHECK:  mov     w7, #8
; ... setup call with swiferror %error_ptr_ref.
; CHECK:  ldr     x21, [sp, #8]
; CHECK:  bl      _params_in_reg2
; Restore return values for return from this function.
; CHECK:  mov      x0, x20
; CHECK:  mov      x1, x22
; CHECK:  mov      x2, x23
; CHECK:  mov      x3, x24
; CHECK:  mov      x4, x25
; CHECK:  mov      x5, x26
; CHECK:  mov      x6, x27
; CHECK:  mov      x7, x28
; CHECK:  mov      x21, x19
; Restore callee save registers.
; CHECK:  ldp     x29, x30, [sp
; CHECK:  ldp     x20, x19, [sp
; CHECK:  ldp     x23, x22, [sp
; CHECK:  ldp     x25, x24, [sp
; CHECK:  ldp     x27, x26, [sp
; CHECK:  ldr     x28, [sp
; CHECK:  ret
define swiftcc { i64, i64, i64, i64, i64, i64, i64, i64 } @params_and_return_in_reg(i64, i64, i64, i64, i64, i64, i64, i64, i8* , %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i8*  null, %swift_error** nocapture swifterror %error_ptr_ref)
  %val = call swiftcc  { i64, i64, i64, i64, i64, i64, i64, i64 } @params_and_return_in_reg2(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i8*  %8, %swift_error** nocapture swifterror %err)
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i8*  null, %swift_error** nocapture swifterror %error_ptr_ref)
  ret { i64, i64, i64, i64, i64, i64, i64, i64 } %val
}

declare swiftcc { i64, i64, i64, i64, i64, i64, i64, i64 } @params_and_return_in_reg2(i64, i64, i64, i64, i64, i64, i64, i64, i8* , %swift_error** nocapture swifterror %err)

declare void @acallee(i8*)

; Make sure we don't tail call if the caller returns a swifterror value. We
; would have to move into the swifterror register before the tail call.
; CHECK: tailcall_from_swifterror:
; CHECK-NOT: b _acallee
; CHECK: bl _acallee

define swiftcc void @tailcall_from_swifterror(%swift_error** swifterror %error_ptr_ref) {
entry:
  tail call void @acallee(i8* null)
  ret void
}

; CHECK: tailcall_from_swifterror2
; CHECK-NOT: b _simple_fn
; CHECK: bl _simple_fn
declare void @simple_fn()
define swiftcc void @tailcall_from_swifterror2(%swift_error** swifterror %error_ptr_ref) {
  tail call void @simple_fn()
  ret void
}

declare swiftcc void @foo2(%swift_error** swifterror)
; CHECK-LABEL: testAssign
; CHECK: mov      x21, xzr
; CHECK: bl      _foo2
; CHECK: mov      x0, x21

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

; foo takes a swifterror parameter. We should be able to see that even when
; it isn't explicitly on the call.
define float @swifterror_param_not_on_call(i8* %error_ref) {
; CHECK-LABEL: swifterror_param_not_on_call:
; CHECK: mov [[ID:x[0-9]+]], x0
; CHECK: bl {{.*}}foo
; CHECK: mov x0, x21
; CHECK: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK: bl {{.*}}free

entry:
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  %call = call float @foo(%swift_error** %error_ptr_ref)
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

; foo_sret takes an sret parameter and a swifterror parameter. We should be
; able to see that, even if it's not explicitly on the call.
define float @swifterror_param_not_on_call2(i8* %error_ref) {
; CHECK-LABEL: swifterror_param_not_on_call2:
; CHECK: mov [[ID:x[0-9]+]], x0
; CHECK: mov [[ZERO:x[0-9]+]], xzr
; CHECK: bl {{.*}}foo_sret
; CHECK: mov x0, x21
; CHECK: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK: bl {{.*}}free

entry:
  %s = alloca %struct.S, align 8
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  call void @foo_sret(%struct.S* %s, i32 1, %swift_error** %error_ptr_ref)
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
