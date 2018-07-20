; RUN: llc -fast-isel-sink-local-values -verify-machineinstrs < %s -mtriple=armv7-apple-ios | FileCheck --check-prefix=CHECK-APPLE --check-prefix=CHECK-ARMV7 %s
; RUN: llc -fast-isel-sink-local-values -verify-machineinstrs -O0 < %s -mtriple=armv7-apple-ios | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -fast-isel-sink-local-values -verify-machineinstrs < %s -mtriple=armv7-linux-androideabi | FileCheck --check-prefix=CHECK-ANDROID %s

declare i8* @malloc(i64)
declare void @free(i8*)
%swift_error = type { i64, i8 }
%struct.S = type { i32, i32, i32, i32, i32, i32 }

; This tests the basic usage of a swifterror parameter. "foo" is the function
; that takes a swifterror parameter and "caller" is the caller of "foo".
define float @foo(%swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: foo:
; CHECK-APPLE: mov r0, #16
; CHECK-APPLE: malloc
; CHECK-APPLE-DAG: mov [[ID:r[0-9]+]], #1
; CHECK-APPLE-DAG: mov r8, r{{.*}}
; CHECK-APPLE-DAG: strb [[ID]], [r{{.*}}, #8]

; CHECK-O0-LABEL: foo:
; CHECK-O0: mov r{{.*}}, #16
; CHECK-O0: malloc
; CHECK-O0: mov [[ID2:r[0-9]+]], r0
; CHECK-O0: mov [[ID:r[0-9]+]], #1
; CHECK-O0: strb [[ID]], [r0, #8]
; CHECK-O0: mov r8, [[ID2]]
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
; CHECK-APPLE-DAG: mov [[ID:r[0-9]+]], r0
; CHECK-APPLE-DAG: mov r8, #0
; CHECK-APPLE: bl {{.*}}foo
; CHECK-APPLE: mov r0, r8
; CHECK-APPLE: cmp r8, #0
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrbeq [[CODE:r[0-9]+]], [r0, #8]
; CHECK-APPLE: strbeq [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller:
; spill r0
; CHECK-O0-DAG: mov r8, #0
; CHECK-O0-DAG: str r0, [sp, [[SLOT:#[0-9]+]]
; CHECK-O0: bl {{.*}}foo
; CHECK-O0: mov [[TMP:r[0-9]+]], r8
; CHECK-O0: str [[TMP]], [sp]
; CHECK-O0: bne
; CHECK-O0: ldrb [[CODE:r[0-9]+]], [r0, #8]
; CHECK-O0: ldr     [[ID:r[0-9]+]], [sp, [[SLOT]]]
; CHECK-O0: strb [[CODE]], [{{.*}}[[ID]]]
; reload r0
; CHECK-O0: ldr r0, [sp]
; CHECK-O0: free
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
; CHECK-APPLE-DAG: mov [[ID:r[0-9]+]], r0
; CHECK-APPLE-DAG: mov r8, #0
; CHECK-APPLE: bl {{.*}}foo
; CHECK-APPLE: cmp r8, #0
; CHECK-APPLE: bne
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:r[0-9]+]], [r8, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: mov r0, r8
; CHECK-APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller2:
; spill r0
; CHECK-O0-DAG: str r0,
; CHECK-O0-DAG: mov r8, #0
; CHECK-O0: bl {{.*}}foo
; CHECK-O0: mov r{{.*}}, r8
; CHECK-O0: str r0, [sp]
; CHECK-O0: bne
; CHECK-O0: ble
; CHECK-O0: ldrb [[CODE:r[0-9]+]], [r0, #8]
; reload r0
; CHECK-O0: ldr [[ID:r[0-9]+]],
; CHECK-O0: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-O0: ldr r0, [sp]
; CHECK-O0: free
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
; CHECK-APPLE: cmp r0, #0
; CHECK-APPLE: eq
; CHECK-APPLE: mov r0, #16
; CHECK-APPLE: malloc
; CHECK-APPLE-DAG: mov [[ID:r[0-9]+]], #1
; CHECK-APPLE-DAG: mov r8, r{{.*}}
; CHECK-APPLE-DAG: strb [[ID]], [r{{.*}}, #8]

; CHECK-O0-LABEL: foo_if:
; CHECK-O0: cmp r0, #0
; spill to stack
; CHECK-O0: str r8
; CHECK-O0: beq
; CHECK-O0: mov r0, #16
; CHECK-O0: malloc
; CHECK-O0: mov [[ID:r[0-9]+]], r0
; CHECK-O0: mov [[ID2:[a-z0-9]+]], #1
; CHECK-O0: strb [[ID2]], [r0, #8]
; CHECK-O0: mov r8, [[ID]]
; reload from stack
; CHECK-O0: ldr r8
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
; CHECK-APPLE: mov [[CODE:r[0-9]+]], r0
; swifterror is kept in a register
; CHECK-APPLE: cmp [[CODE]], #0
; CHECK-APPLE: beq
; CHECK-APPLE: mov r0, #16
; CHECK-APPLE: malloc
; CHECK-APPLE: strb r{{.*}}, [r0, #8]
; CHECK-APPLE: ble

; CHECK-O0-LABEL: foo_loop:
; CHECK-O0: mov r{{.*}}, r8
; CHECK-O0: cmp r{{.*}}, #0
; CHECK-O0: beq
; CHECK-O0: mov r0, #16
; CHECK-O0: malloc
; CHECK-O0-DAG: mov [[ID:r[0-9]+]], r0
; CHECK-O0-DAG: movw [[ID2:.*]], #1
; CHECK-O0: strb [[ID2]], [{{.*}}[[ID]], #8]
; spill r0
; CHECK-O0: str r0, [sp{{.*}}]
; CHECK-O0: vcmpe
; CHECK-O0: ble
; reload from stack
; CHECK-O0: ldr r8
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

; "foo_sret" is a function that takes a swifterror parameter, it also has a sret
; parameter.
define void @foo_sret(%struct.S* sret %agg.result, i32 %val1, %swift_error** swifterror %error_ptr_ref) {
; CHECK-APPLE-LABEL: foo_sret:
; CHECK-APPLE: mov [[SRET:r[0-9]+]], r0
; CHECK-APPLE: mov r0, #16
; CHECK-APPLE: malloc
; CHECK-APPLE: mov [[REG:r[0-9]+]], #1
; CHECK-APPLE-DAG: mov r8, r0
; CHECK-APPLE-DAG: strb [[REG]], [r0, #8]
; CHECK-APPLE-DAG: str r{{.*}}, [{{.*}}[[SRET]], #4]

; CHECK-O0-LABEL: foo_sret:
; CHECK-O0: mov r{{.*}}, #16
; spill to stack: sret and val1
; CHECK-O0-DAG: str r0
; CHECK-O0-DAG: str r1
; CHECK-O0: malloc
; CHECK-O0: mov [[ID:r[0-9]+]], #1
; CHECK-O0: strb [[ID]], [r0, #8]
; reload from stack: sret and val1
; CHECK-O0: ldr
; CHECK-O0: ldr
; CHECK-O0: str r{{.*}}, [{{.*}}, #4]
; CHECK-O0: mov r8
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
; CHECK-APPLE: mov [[ID:r[0-9]+]], r0
; CHECK-APPLE: mov r8, #0
; CHECK-APPLE: bl {{.*}}foo_sret
; CHECK-APPLE: mov r0, r8
; CHECK-APPLE: cmp r8, #0
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrbeq [[CODE:r[0-9]+]], [r0, #8]
; CHECK-APPLE: strbeq [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller3:
; CHECK-O0-DAG: mov r8, #0
; CHECK-O0-DAG: mov r0
; CHECK-O0-DAG: mov r1
; CHECK-O0: bl {{.*}}foo_sret
; CHECK-O0: mov [[ID2:r[0-9]+]], r8
; CHECK-O0: cmp r8
; CHECK-O0: str [[ID2]], [sp[[SLOT:.*]]]
; CHECK-O0: bne
; Access part of the error object and save it to error_ref
; CHECK-O0: ldrb [[CODE:r[0-9]+]]
; CHECK-O0: ldr [[ID:r[0-9]+]]
; CHECK-O0: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-O0: ldr r0, [sp[[SLOT]]
; CHECK-O0: bl {{.*}}free
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
; CHECK-APPLE: mov r0, #16
; CHECK-APPLE: malloc
; CHECK-APPLE: mov r8, r0
; CHECK-APPLE: mov [[ID:r[0-9]+]], #1
; CHECK-APPLE-DAG: strb [[ID]], [r8, #8]

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
; CHECK-APPLE: mov [[ID:r[0-9]+]], r0
; CHECK-APPLE: mov r8, #0
; CHECK-APPLE: bl {{.*}}foo_vararg
; CHECK-APPLE: mov r0, r8
; CHECK-APPLE: cmp r8, #0
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrbeq [[CODE:r[0-9]+]], [r0, #8]
; CHECK-APPLE: strbeq [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: bl {{.*}}free
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

; CHECK-APPLE-LABEL: swifterror_clobber
; CHECK-APPLE: mov [[REG:r[0-9]+]], r8
; CHECK-APPLE: nop
; CHECK-APPLE: mov r8, [[REG]]
define swiftcc void @swifterror_clobber(%swift_error** nocapture swifterror %err) {
  call void asm sideeffect "nop", "~{r8}"()
  ret void
}

; CHECK-APPLE-LABEL: swifterror_reg_clobber
; CHECK-APPLE: push {{.*}}r8
; CHECK-APPLE: nop
; CHECK-APPLE: pop  {{.*}}r8
define swiftcc void @swifterror_reg_clobber(%swift_error** nocapture %err) {
  call void asm sideeffect "nop", "~{r8}"()
  ret void
}

; CHECK-ARMV7-LABEL: _params_in_reg
; Store callee saved registers excluding swifterror.
; CHECK-ARMV7:  push   {r4, r5, r6, r7, r10, r11, lr}
; Store swiftself (r10) and swifterror (r8).
; CHECK-ARMV7-DAG:  str     r8, [s[[STK1:.*]]]
; CHECK-ARMV7-DAG:  str     r10, [s[[STK2:.*]]]
; Store arguments.
; CHECK-ARMV7-DAG:  mov     r6, r3
; CHECK-ARMV7-DAG:  mov     r4, r2
; CHECK-ARMV7-DAG:  mov     r11, r1
; CHECK-ARMV7-DAG:  mov     r5, r0
; Setup call.
; CHECK-ARMV7:  mov     r0, #1
; CHECK-ARMV7:  mov     r1, #2
; CHECK-ARMV7:  mov     r2, #3
; CHECK-ARMV7:  mov     r3, #4
; CHECK-ARMV7:  mov     r10, #0
; CHECK-ARMV7:  mov     r8, #0
; CHECK-ARMV7:  bl      _params_in_reg2
; Restore original arguments.
; CHECK-ARMV7-DAG:  ldr     r10, [s[[STK2]]]
; CHECK-ARMV7-DAG:  ldr     r8, [s[[STK1]]]
; CHECK-ARMV7-DAG:  mov     r0, r5
; CHECK-ARMV7-DAG:  mov     r1, r11
; CHECK-ARMV7-DAG:  mov     r2, r4
; CHECK-ARMV7-DAG:  mov     r3, r6
; CHECK-ARMV7:  bl      _params_in_reg2
; CHECK-ARMV7:  pop     {r4, r5, r6, r7, r10, r11, pc}
define swiftcc void @params_in_reg(i32, i32, i32, i32, i8* swiftself, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i32 1, i32 2, i32 3, i32 4, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  call swiftcc void @params_in_reg2(i32 %0, i32 %1, i32 %2, i32 %3, i8* swiftself %4, %swift_error** nocapture swifterror %err)
  ret void
}
declare swiftcc void @params_in_reg2(i32, i32, i32, i32, i8* swiftself, %swift_error** nocapture swifterror %err)

; CHECK-ARMV7-LABEL: params_and_return_in_reg
; CHECK-ARMV7:  push    {r4, r5, r6, r7, r10, r11, lr}
; Store swifterror and swiftself
; CHECK-ARMV7:  mov     r6, r8
; CHECK-ARMV7:  str     r10, [s[[STK1:.*]]]
; Store arguments.
; CHECK-ARMV7:  str     r3, [s[[STK2:.*]]]
; CHECK-ARMV7:  mov     r4, r2
; CHECK-ARMV7:  mov     r11, r1
; CHECK-ARMV7:  mov     r5, r0
; Setup call.
; CHECK-ARMV7:  mov     r0, #1
; CHECK-ARMV7:  mov     r1, #2
; CHECK-ARMV7:  mov     r2, #3
; CHECK-ARMV7:  mov     r3, #4
; CHECK-ARMV7:  mov     r10, #0
; CHECK-ARMV7:  mov     r8, #0
; CHECK-ARMV7:  bl      _params_in_reg2
; Restore original arguments.
; CHECK-ARMV7-DAG:  ldr     r3, [s[[STK2]]]
; CHECK-ARMV7-DAG:  ldr     r10, [s[[STK1]]]
; Store %error_ptr_ref;
; CHECK-ARMV7-DAG:  str     r8, [s[[STK3:.*]]]
; Restore original arguments.
; CHECK-ARMV7-DAG:  mov     r0, r5
; CHECK-ARMV7-DAG:  mov     r1, r11
; CHECK-ARMV7-DAG:  mov     r2, r4
; CHECK-ARMV7-DAG:  mov     r8, r6
; CHECK-ARMV7:  bl      _params_and_return_in_reg2
; Store swifterror return %err;
; CHECK-ARMV7-DAG:  str     r8, [s[[STK1]]]
; Load swifterror value %error_ptr_ref.
; CHECK-ARMV7-DAG:  ldr     r8, [s[[STK3]]]
; Save return values.
; CHECK-ARMV7-DAG:  mov     r4, r0
; CHECK-ARMV7-DAG:  mov     r5, r1
; CHECK-ARMV7-DAG:  mov     r6, r2
; CHECK-ARMV7-DAG:  mov     r11, r3
; Setup call.
; CHECK-ARMV7:  mov     r0, #1
; CHECK-ARMV7:  mov     r1, #2
; CHECK-ARMV7:  mov     r2, #3
; CHECK-ARMV7:  mov     r3, #4
; CHECK-ARMV7:  mov     r10, #0
; CHECK-ARMV7:  bl      _params_in_reg2
; Load swifterror %err;
; CHECK-ARMV7-DAG:  ldr     r8, [s[[STK1]]]
; Restore return values for returning.
; CHECK-ARMV7-DAG:  mov     r0, r4
; CHECK-ARMV7-DAG:  mov     r1, r5
; CHECK-ARMV7-DAG:  mov     r2, r6
; CHECK-ARMV7-DAG:  mov     r3, r11
; CHECK-ARMV7:  pop     {r4, r5, r6, r7, r10, r11, pc}

; CHECK-ANDROID-LABEL: params_and_return_in_reg
; CHECK-ANDROID:  push    {r4, r5, r6, r7, r9, r10, r11, lr}
; CHECK-ANDROID:  sub     sp, sp, #16
; CHECK-ANDROID:  str     r8, [sp, #4]            @ 4-byte Spill
; CHECK-ANDROID:  mov     r11, r10
; CHECK-ANDROID:  mov     r6, r3
; CHECK-ANDROID:  mov     r7, r2
; CHECK-ANDROID:  mov     r4, r1
; CHECK-ANDROID:  mov     r5, r0
; CHECK-ANDROID:  mov     r0, #1
; CHECK-ANDROID:  mov     r1, #2
; CHECK-ANDROID:  mov     r2, #3
; CHECK-ANDROID:  mov     r3, #4
; CHECK-ANDROID:  mov     r10, #0
; CHECK-ANDROID:  mov     r8, #0
; CHECK-ANDROID:  bl      params_in_reg2
; CHECK-ANDROID:  mov     r9, r8
; CHECK-ANDROID:  ldr     r8, [sp, #4]            @ 4-byte Reload
; CHECK-ANDROID:  mov     r0, r5
; CHECK-ANDROID:  mov     r1, r4
; CHECK-ANDROID:  mov     r2, r7
; CHECK-ANDROID:  mov     r3, r6
; CHECK-ANDROID:  mov     r10, r11
; CHECK-ANDROID:  bl      params_and_return_in_reg2
; CHECK-ANDROID:  mov     r4, r0
; CHECK-ANDROID:  mov     r5, r1
; CHECK-ANDROID:  mov     r6, r2
; CHECK-ANDROID:  mov     r7, r3
; CHECK-ANDROID:  mov     r11, r8
; CHECK-ANDROID:  mov     r0, #1
; CHECK-ANDROID:  mov     r1, #2
; CHECK-ANDROID:  mov     r2, #3
; CHECK-ANDROID:  mov     r3, #4
; CHECK-ANDROID:  mov     r10, #0
; CHECK-ANDROID:  mov     r8, r9
; CHECK-ANDROID:  bl      params_in_reg2
; CHECK-ANDROID:  mov     r0, r4
; CHECK-ANDROID:  mov     r1, r5
; CHECK-ANDROID:  mov     r2, r6
; CHECK-ANDROID:  mov     r3, r7
; CHECK-ANDROID:  mov     r8, r11
; CHECK-ANDROID:  add     sp, sp, #16
; CHECK-ANDROID:  pop	{r4, r5, r6, r7, r9, r10, r11, pc}

define swiftcc { i32, i32, i32, i32} @params_and_return_in_reg(i32, i32, i32, i32, i8* swiftself, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i32 1, i32 2, i32 3, i32 4, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  %val = call swiftcc  { i32, i32, i32, i32 } @params_and_return_in_reg2(i32 %0, i32 %1, i32 %2, i32 %3, i8* swiftself %4, %swift_error** nocapture swifterror %err)
  call swiftcc void @params_in_reg2(i32 1, i32 2, i32 3, i32 4, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  ret { i32, i32, i32, i32 }%val
}

declare swiftcc { i32, i32, i32, i32 } @params_and_return_in_reg2(i32, i32, i32, i32, i8* swiftself, %swift_error** nocapture swifterror %err)


declare void @acallee(i8*)

; Make sure we don't tail call if the caller returns a swifterror value. We
; would have to move into the swifterror register before the tail call.
; CHECK-APPLE: tailcall_from_swifterror:
; CHECK-APPLE-NOT: b _acallee
; CHECK-APPLE: bl _acallee
; CHECK-ANDROID: tailcall_from_swifterror:
; CHECK-ANDROID-NOT: b acallee
; CHECK-ANDROID: bl acallee

define swiftcc void @tailcall_from_swifterror(%swift_error** swifterror %error_ptr_ref) {
entry:
  tail call void @acallee(i8* null)
  ret void
}


declare swiftcc void @foo2(%swift_error** swifterror)

; Make sure we properly assign registers during fast-isel.
; CHECK-O0-LABEL: testAssign
; CHECK-O0: mov     r8, #0
; CHECK-O0: bl      _foo2
; CHECK-O0: str     r8, [s[[STK:p.*]]]
; CHECK-O0: ldr     r0, [s[[STK]]]
; CHECK-O0: pop

; CHECK-APPLE-LABEL: testAssign
; CHECK-APPLE:  mov     r8, #0
; CHECK-APPLE:  bl      _foo2
; CHECK-APPLE:  mov     r0, r8

; CHECK-ANDROID-LABEL: testAssign
; CHECK-ANDROID:  mov     r8, #0
; CHECK-ANDROID:  bl      foo2
; CHECK-ANDROID:  mov     r0, r8

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
