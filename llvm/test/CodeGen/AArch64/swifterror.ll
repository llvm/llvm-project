; RUN: llc -verify-machineinstrs -disable-fp-elim < %s -mtriple=aarch64-apple-ios -disable-post-ra | FileCheck --check-prefix=CHECK-APPLE %s
; RUN: llc -verify-machineinstrs -disable-fp-elim -O0 < %s -mtriple=aarch64-apple-ios -disable-post-ra | FileCheck --check-prefix=CHECK-O0 %s

declare i8* @malloc(i64)
declare void @free(i8*)
%swift_error = type {i64, i8}

; This tests the basic usage of a swifterror parameter. "foo" is the function
; that takes a swifterror parameter and "caller" is the caller of "foo".
define float @foo(%swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: foo:
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-APPLE: strb [[ID]], [x0, #8]
; CHECK-APPLE: mov x19, x0
; CHECK-APPLE-NOT: x19

; CHECK-O0-LABEL: foo:
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; CHECK-O0: malloc
; CHECK-O0: mov [[ID2:x[0-9]+]], x0
; CHECK-O0: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-O0: strb [[ID]], [x0, #8]
; CHECK-O0: mov x19, [[ID2]]
; CHECK-O0-NOT: x19
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
; CHECK-APPLE: mov [[ID:x[0-9]+]], x0
; CHECK-APPLE: mov x19, xzr
; CHECK-APPLE: bl {{.*}}foo
; CHECK-APPLE: cbnz x19
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x19, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: mov x0, x19
; CHECK_APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller:
; CHECK-O0: mov x19
; CHECK-O0: bl {{.*}}foo
; CHECK-O0: mov [[ID:x[0-9]+]], x19
; CHECK-O0: cbnz [[ID]]
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
; CHECK-APPLE: mov [[ID:x[0-9]+]], x0
; CHECK-APPLE: fmov [[CMP:s[0-9]+]], #1.0
; CHECK-APPLE: mov x19, xzr
; CHECK-APPLE: bl {{.*}}foo
; CHECK-APPLE: cbnz x19
; CHECK-APPLE: fcmp s0, [[CMP]]
; CHECK-APPLE: b.le
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x19, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: mov x0, x19
; CHECK_APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller2:
; CHECK-O0: mov x19
; CHECK-O0: bl {{.*}}foo
; CHECK-O0: mov [[ID:x[0-9]+]], x19
; CHECK-O0: cbnz [[ID]]
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
; CHECK-APPLE: cbz w0
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-APPLE: strb [[ID]], [x0, #8]
; CHECK-APPLE: mov x19, x0
; CHECK-APPLE-NOT: x19
; CHECK-APPLE: ret

; CHECK-O0-LABEL: foo_if:
; spill x19
; CHECK-O0: str x19
; CHECK-O0: cbz w0
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; CHECK-O0: malloc
; CHECK-O0: mov [[ID:x[0-9]+]], x0
; CHECK-O0: orr [[ID2:w[0-9]+]], wzr, #0x1
; CHECK-O0: strb [[ID2]], [x0, #8]
; CHECK-O0: mov x19, [[ID]]
; CHECK-O0: ret
; reload from stack
; CHECK-O0: ldr x19
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
; CHECK-APPLE: mov x0, x19
; CHECK-APPLE: cbz
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE: strb w{{.*}}, [x0, #8]
; CHECK-APPLE: fcmp
; CHECK-APPLE: b.le
; CHECK-APPLE: mov x19, x0
; CHECK-APPLE: ret

; CHECK-O0-LABEL: foo_loop:
; spill x19
; CHECK-O0: str x19
; CHECk-O0: cbz
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; CHECK-O0: malloc
; CHECK-O0: mov [[ID:x[0-9]+]], x0
; CHECK-O0: strb w{{.*}}, [{{.*}}[[ID]], #8]
; spill x0
; CHECK-O0: str x0
; CHECK-O0: fcmp
; CHECK-O0: b.le
; reload from stack
; CHECK-O0: ldr x19
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
; CHECK-APPLE: mov [[SRET:x[0-9]+]], x8
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-APPLE: strb [[ID]], [x0, #8]
; CHECK-APPLE: str w{{.*}}, [{{.*}}[[SRET]], #4]
; CHECK-APPLE: mov x19, x0
; CHECK-APPLE-NOT: x19

; CHECK-O0-LABEL: foo_sret:
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; spill x8
; CHECK-O0-DAG: str x8
; spill x19
; CHECK-O0-DAG: str x19
; CHECK-O0: malloc
; CHECK-O0: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-O0: strb [[ID]], [x0, #8]
; reload from stack
; CHECK-O0: ldr [[SRET:x[0-9]+]]
; CHECK-O0: str w{{.*}}, [{{.*}}[[SRET]], #4]
; CHECK-O0: mov x19
; CHECK-O0-NOT: x19
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
; CHECK-APPLE: mov [[ID:x[0-9]+]], x0
; CHECK-APPLE: mov x19, xzr
; CHECK-APPLE: bl {{.*}}foo_sret
; CHECK-APPLE: cbnz x19
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x19, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: mov x0, x19
; CHECK_APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller3:
; spill x0
; CHECK-O0: str x0
; CHECK-O0: mov x19
; CHECK-O0: bl {{.*}}foo_sret
; CHECK-O0: mov [[ID2:x[0-9]+]], x19
; CHECK-O0: cbnz [[ID2]]
; Access part of the error object and save it to error_ref
; reload from stack
; CHECK-O0: ldrb [[CODE:w[0-9]+]]
; CHECK-O0: ldr [[ID:x[0-9]+]]
; CHECK-O0: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK_O0: bl {{.*}}free
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
; CHECK-APPLE-LABEL: foo_vararg:
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-APPLE: add [[ARGS:x[0-9]+]], [[TMP:x[0-9]+]], #16
; CHECK-APPLE: strb [[ID]], [x0, #8]

; First vararg
; CHECK-APPLE-DAG: orr {{x[0-9]+}}, [[ARGS]], #0x8
; CHECK-APPLE-DAG: ldr {{w[0-9]+}}, [{{.*}}[[TMP]], #16]
; CHECK-APPLE: add {{x[0-9]+}}, {{x[0-9]+}}, #8
; Second vararg
; CHECK-APPLE: ldr {{w[0-9]+}}, [{{x[0-9]+}}]
; CHECK-APPLE: add {{x[0-9]+}}, {{x[0-9]+}}, #8
; Third vararg
; CHECK-APPLE: ldr {{w[0-9]+}}, [{{x[0-9]+}}]

; CHECK-APPLE: mov x19, x0
; CHECK-APPLE-NOT: x19
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
; CHECK-APPLE-LABEL: caller4:

; CHECK-APPLE: mov [[ID:x[0-9]+]], x0
; CHECK-APPLE: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #8]
; CHECK-APPLE: str {{x[0-9]+}}, [sp]

; CHECK-APPLE: mov x19, xzr
; CHECK-APPLE: bl {{.*}}foo_vararg
; CHECK-APPLE: cbnz x19
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x19, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: mov x0, x19
; CHECK_APPLE: bl {{.*}}free
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
