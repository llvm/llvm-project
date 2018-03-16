; RUN: llc -verify-machineinstrs -disable-fp-elim -enable-shrink-wrap=false < %s -mtriple=aarch64-apple-ios -disable-post-ra | FileCheck --check-prefix=CHECK-APPLE %s
; RUN: llc -verify-machineinstrs -disable-fp-elim -O0 -fast-isel < %s -mtriple=aarch64-apple-ios -disable-post-ra | FileCheck --check-prefix=CHECK-O0 %s

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
; CHECK-APPLE: mov x21, x0
; CHECK-APPLE-NOT: x21

; CHECK-O0-LABEL: foo:
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; CHECK-O0: malloc
; CHECK-O0: mov x21, x0
; CHECK-O0-NOT: x21
; CHECK-O0: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-O0-NOT: x21
; CHECK-O0: strb [[ID]], [x0, #8]
; CHECK-O0-NOT: x21
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
; CHECK-APPLE: mov x21, xzr
; CHECK-APPLE: bl {{.*}}foo
; CHECK-APPLE: mov x0, x21
; CHECK-APPLE: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller:
; CHECK-O0: mov x21
; CHECK-O0: bl {{.*}}foo
; CHECK-O0: mov [[ID:x[0-9]+]], x21
; CHECK-O0: cbnz x21
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
; CHECK-APPLE: mov x21, xzr
; CHECK-APPLE: bl {{.*}}foo
; CHECK-APPLE: cbnz x21
; CHECK-APPLE: fcmp s0, [[CMP]]
; CHECK-APPLE: b.le
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x21, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: mov x0, x21
; CHECK-APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller2:
; CHECK-O0: mov x21
; CHECK-O0: bl {{.*}}foo
; CHECK-O0: mov [[ID:x[0-9]+]], x21
; CHECK-O0: cbnz x21
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
; CHECK-APPLE: mov x21, x0
; CHECK-APPLE-NOT: x21
; CHECK-APPLE: ret

; CHECK-O0-LABEL: foo_if:
; spill x21
; CHECK-O0: str x21, [sp, [[SLOT:#[0-9]+]]]
; CHECK-O0: cbz w0
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; CHECK-O0: malloc
; CHECK-O0: mov [[ID:x[0-9]+]], x0
; CHECK-O0: orr [[ID2:w[0-9]+]], wzr, #0x1
; CHECK-O0: strb [[ID2]], [x0, #8]
; CHECK-O0: mov x21, [[ID]]
; CHECK-O0: ret
; reload from stack
; CHECK-O0: ldr [[ID3:x[0-9]+]], [sp, [[SLOT]]]
; CHECK-O0: mov x21, [[ID3]]
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
; CHECK-APPLE: mov x0, x21
; CHECK-APPLE: cbz
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE: strb w{{.*}}, [x0, #8]
; CHECK-APPLE: fcmp
; CHECK-APPLE: b.le
; CHECK-APPLE: mov x21, x0
; CHECK-APPLE: ret

; CHECK-O0-LABEL: foo_loop:
; spill x21
; CHECK-O0: str x21, [sp, [[SLOT:#[0-9]+]]]
; CHECK-O0: b [[BB1:[A-Za-z0-9_]*]]
; CHECK-O0: [[BB1]]:
; CHECK-O0: ldr     x0, [sp, [[SLOT]]]
; CHECK-O0: str     x0, [sp, [[SLOT2:#[0-9]+]]]
; CHECK-O0: cbz {{.*}}, [[BB2:[A-Za-z0-9_]*]]
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; CHECK-O0: malloc
; CHECK-O0: mov [[ID:x[0-9]+]], x0
; CHECK-O0: strb w{{.*}}, [{{.*}}[[ID]], #8]
; spill x0
; CHECK-O0: str x0, [sp, [[SLOT2]]]
; CHECK-O0:[[BB2]]:
; CHECK-O0: ldr     x0, [sp, [[SLOT2]]]
; CHECK-O0: fcmp
; CHECK-O0: str     x0, [sp, [[SLOT3:#[0-9]+]]
; CHECK-O0: b.le [[BB1]]
; reload from stack
; CHECK-O0: ldr [[ID3:x[0-9]+]], [sp, [[SLOT3]]]
; CHECK-O0: mov x21, [[ID3]]
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
; CHECK-APPLE: mov x21, x0
; CHECK-APPLE-NOT: x21

; CHECK-O0-LABEL: foo_sret:
; CHECK-O0: orr w{{.*}}, wzr, #0x10
; spill x8
; CHECK-O0-DAG: str x8
; spill x21
; CHECK-O0-DAG: str x21
; CHECK-O0: malloc
; CHECK-O0: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-O0: strb [[ID]], [x0, #8]
; reload from stack
; CHECK-O0: ldr [[SRET:x[0-9]+]]
; CHECK-O0: str w{{.*}}, [{{.*}}[[SRET]], #4]
; CHECK-O0: mov x21
; CHECK-O0-NOT: x21
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
; CHECK-APPLE: mov x21, xzr
; CHECK-APPLE: bl {{.*}}foo_sret
; CHECK-APPLE: mov x0, x21
; CHECK-APPLE: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
; CHECK-APPLE: bl {{.*}}free

; CHECK-O0-LABEL: caller3:
; spill x0
; CHECK-O0: str x0
; CHECK-O0: mov x21
; CHECK-O0: bl {{.*}}foo_sret
; CHECK-O0: mov [[ID2:x[0-9]+]], x21
; CHECK-O0: cbnz x21
; Access part of the error object and save it to error_ref
; reload from stack
; CHECK-O0: ldrb [[CODE:w[0-9]+]]
; CHECK-O0: ldr [[ID:x[0-9]+]]
; CHECK-O0: strb [[CODE]], [{{.*}}[[ID]]]
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
; CHECK-APPLE: orr w0, wzr, #0x10
; CHECK-APPLE: malloc
; CHECK-APPLE-DAG: orr [[ID:w[0-9]+]], wzr, #0x1
; CHECK-APPLE-DAG: add [[ARGS:x[0-9]+]], [[TMP:x[0-9]+]], #16
; CHECK-APPLE-DAG: strb [[ID]], [x0, #8]

; First vararg
; CHECK-APPLE-DAG: orr {{x[0-9]+}}, [[ARGS]], #0x8
; CHECK-APPLE-DAG: ldr {{w[0-9]+}}, [{{.*}}[[TMP]], #16]
; Second vararg
; CHECK-APPLE-DAG: ldr {{w[0-9]+}}, [{{x[0-9]+}}], #8
; CHECK-APPLE-DAG: add {{x[0-9]+}}, {{x[0-9]+}}, #16
; Third vararg
; CHECK-APPLE: ldr {{w[0-9]+}}, [{{x[0-9]+}}], #8

; CHECK-APPLE: mov x21, x0
; CHECK-APPLE-NOT: x21
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

; CHECK-APPLE: mov x21, xzr
; CHECK-APPLE: bl {{.*}}foo_vararg
; CHECK-APPLE: mov x0, x21
; CHECK-APPLE: cbnz x21
; Access part of the error object and save it to error_ref
; CHECK-APPLE: ldrb [[CODE:w[0-9]+]], [x0, #8]
; CHECK-APPLE: strb [[CODE]], [{{.*}}[[ID]]]
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
; CHECK-APPLE: mov [[REG:x[0-9]+]], x21
; CHECK-APPLE: nop
; CHECK-APPLE: mov x21, [[REG]]
define swiftcc void @swifterror_clobber(%swift_error** nocapture swifterror %err) {
  call void asm sideeffect "nop", "~{x21}"()
  ret void
}

; CHECK-APPLE-LABEL: swifterror_reg_clobber
; CHECK-APPLE: stp {{.*}}x21
; CHECK-APPLE: nop
; CHECK-APPLE: ldp  {{.*}}x21
define swiftcc void @swifterror_reg_clobber(%swift_error** nocapture %err) {
  call void asm sideeffect "nop", "~{x21}"()
  ret void
}
; CHECK-APPLE-LABEL: params_in_reg
; Save callee saved registers and swifterror since it will be clobbered by the first call to params_in_reg2.
; CHECK-APPLE:  stp     x21, x28, [sp
; CHECK-APPLE:  stp     x27, x26, [sp
; CHECK-APPLE:  stp     x25, x24, [sp
; CHECK-APPLE:  stp     x23, x22, [sp
; CHECK-APPLE:  stp     x20, x19, [sp
; CHECK-APPLE:  stp     x29, x30, [sp
; CHECK-APPLE:  str     x20, [sp
; Store argument registers.
; CHECK-APPLE:  mov      x23, x7
; CHECK-APPLE:  mov      x24, x6
; CHECK-APPLE:  mov      x25, x5
; CHECK-APPLE:  mov      x26, x4
; CHECK-APPLE:  mov      x27, x3
; CHECK-APPLE:  mov      x28, x2
; CHECK-APPLE:  mov      x19, x1
; CHECK-APPLE:  mov      x22, x0
; Setup call.
; CHECK-APPLE:  orr     w0, wzr, #0x1
; CHECK-APPLE:  orr     w1, wzr, #0x2
; CHECK-APPLE:  orr     w2, wzr, #0x3
; CHECK-APPLE:  orr     w3, wzr, #0x4
; CHECK-APPLE:  mov     w4, #5
; CHECK-APPLE:  orr     w5, wzr, #0x6
; CHECK-APPLE:  orr     w6, wzr, #0x7
; CHECK-APPLE:  orr     w7, wzr, #0x8
; CHECK-APPLE:  mov      x20, xzr
; CHECK-APPLE:  mov      x21, xzr
; CHECK-APPLE:  bl      _params_in_reg2
; Restore original arguments for next call.
; CHECK-APPLE:  mov      x0, x22
; CHECK-APPLE:  mov      x1, x19
; CHECK-APPLE:  mov      x2, x28
; CHECK-APPLE:  mov      x3, x27
; CHECK-APPLE:  mov      x4, x26
; CHECK-APPLE:  mov      x5, x25
; CHECK-APPLE:  mov      x6, x24
; CHECK-APPLE:  mov      x7, x23
; Restore original swiftself argument and swifterror %err.
; CHECK-APPLE:  ldp             x20, x21, [sp
; CHECK-APPLE:  bl      _params_in_reg2
; Restore calle save registers but don't clober swifterror x21.
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ldp     x29, x30, [sp
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ldp     x20, x19, [sp
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ldp     x23, x22, [sp
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ldp     x25, x24, [sp
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ldp     x27, x26, [sp
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ldr     x28, [sp
; CHECK-APPLE-NOT: x21
; CHECK-APPLE:  ret
define swiftcc void @params_in_reg(i64, i64, i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  call swiftcc void @params_in_reg2(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i8* swiftself %8, %swift_error** nocapture swifterror %err)
  ret void
}
declare swiftcc void @params_in_reg2(i64, i64, i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err)

; CHECK-APPLE-LABEL: params_and_return_in_reg
; Store callee saved registers.
; CHECK-APPLE:  stp     x20, x28, [sp, #24
; CHECK-APPLE:  stp     x27, x26, [sp
; CHECK-APPLE:  stp     x25, x24, [sp
; CHECK-APPLE:  stp     x23, x22, [sp
; CHECK-APPLE:  stp     x20, x19, [sp
; CHECK-APPLE:  stp     x29, x30, [sp
; Save original arguments.
; CHECK-APPLE:  mov      x23, x21
; CHECK-APPLE:  str     x7, [sp, #16]
; CHECK-APPLE:  mov      x24, x6
; CHECK-APPLE:  mov      x25, x5
; CHECK-APPLE:  mov      x26, x4
; CHECK-APPLE:  mov      x27, x3
; CHECK-APPLE:  mov      x28, x2
; CHECK-APPLE:  mov      x19, x1
; CHECK-APPLE:  mov      x22, x0
; Setup call arguments.
; CHECK-APPLE:  orr     w0, wzr, #0x1
; CHECK-APPLE:  orr     w1, wzr, #0x2
; CHECK-APPLE:  orr     w2, wzr, #0x3
; CHECK-APPLE:  orr     w3, wzr, #0x4
; CHECK-APPLE:  mov     w4, #5
; CHECK-APPLE:  orr     w5, wzr, #0x6
; CHECK-APPLE:  orr     w6, wzr, #0x7
; CHECK-APPLE:  orr     w7, wzr, #0x8
; CHECK-APPLE:  mov      x20, xzr
; CHECK-APPLE:  mov      x21, xzr
; CHECK-APPLE:  bl      _params_in_reg2
; Store swifterror %error_ptr_ref.
; CHECK-APPLE:  str     x21, [sp, #8]
; Setup call arguments from original arguments.
; CHECK-APPLE:  mov      x0, x22
; CHECK-APPLE:  mov      x1, x19
; CHECK-APPLE:  mov      x2, x28
; CHECK-APPLE:  mov      x3, x27
; CHECK-APPLE:  mov      x4, x26
; CHECK-APPLE:  mov      x5, x25
; CHECK-APPLE:  mov      x6, x24
; CHECK-APPLE:  ldp     x7, x20, [sp, #16]
; CHECK-APPLE:  mov      x21, x23
; CHECK-APPLE:  bl      _params_and_return_in_reg2
; Store return values.
; CHECK-APPLE:  mov      x19, x0
; CHECK-APPLE:  mov      x22, x1
; CHECK-APPLE:  mov      x24, x2
; CHECK-APPLE:  mov      x25, x3
; CHECK-APPLE:  mov      x26, x4
; CHECK-APPLE:  mov      x27, x5
; CHECK-APPLE:  mov      x28, x6
; CHECK-APPLE:  mov      x23, x7
; Save swifterror %err.
; CHECK-APPLE:  str     x21, [sp, #24]
; Setup call.
; CHECK-APPLE:  orr     w0, wzr, #0x1
; CHECK-APPLE:  orr     w1, wzr, #0x2
; CHECK-APPLE:  orr     w2, wzr, #0x3
; CHECK-APPLE:  orr     w3, wzr, #0x4
; CHECK-APPLE:  mov     w4, #5
; CHECK-APPLE:  orr     w5, wzr, #0x6
; CHECK-APPLE:  orr     w6, wzr, #0x7
; CHECK-APPLE:  orr     w7, wzr, #0x8
; CHECK-APPLE:  mov      x20, xzr
; ... setup call with swiferror %error_ptr_ref.
; CHECK-APPLE:  ldr     x21, [sp, #8]
; CHECK-APPLE:  bl      _params_in_reg2
; Restore return values for return from this function.
; CHECK-APPLE:  mov      x0, x19
; CHECK-APPLE:  mov      x1, x22
; CHECK-APPLE:  mov      x2, x24
; CHECK-APPLE:  mov      x3, x25
; CHECK-APPLE:  mov      x4, x26
; CHECK-APPLE:  mov      x5, x27
; CHECK-APPLE:  mov      x6, x28
; CHECK-APPLE:  mov      x7, x23
; Restore swifterror %err and callee save registers.
; CHECK-APPLE:  ldp     x21, x28, [sp, #24
; CHECK-APPLE:  ldp     x29, x30, [sp
; CHECK-APPLE:  ldp     x20, x19, [sp
; CHECK-APPLE:  ldp     x23, x22, [sp
; CHECK-APPLE:  ldp     x25, x24, [sp
; CHECK-APPLE:  ldp     x27, x26, [sp
; CHECK-APPLE:  ret
define swiftcc { i64, i64, i64, i64, i64, i64, i64, i64 } @params_and_return_in_reg(i64, i64, i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err) {
  %error_ptr_ref = alloca swifterror %swift_error*, align 8
  store %swift_error* null, %swift_error** %error_ptr_ref
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  %val = call swiftcc  { i64, i64, i64, i64, i64, i64, i64, i64 } @params_and_return_in_reg2(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i8* swiftself %8, %swift_error** nocapture swifterror %err)
  call swiftcc void @params_in_reg2(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i8* swiftself null, %swift_error** nocapture swifterror %error_ptr_ref)
  ret { i64, i64, i64, i64, i64, i64, i64, i64 } %val
}

declare swiftcc { i64, i64, i64, i64, i64, i64, i64, i64 } @params_and_return_in_reg2(i64, i64, i64, i64, i64, i64, i64, i64, i8* swiftself, %swift_error** nocapture swifterror %err)

declare void @acallee(i8*)

; Make sure we don't tail call if the caller returns a swifterror value. We
; would have to move into the swifterror register before the tail call.
; CHECK-APPLE: tailcall_from_swifterror:
; CHECK-APPLE-NOT: b _acallee
; CHECK-APPLE: bl _acallee

define swiftcc void @tailcall_from_swifterror(%swift_error** swifterror %error_ptr_ref) {
entry:
  tail call void @acallee(i8* null)
  ret void
}

declare swiftcc void @foo2(%swift_error** swifterror)

; Make sure we properly assign registers during fast-isel.
; CHECK-O0-LABEL: testAssign
; CHECK-O0: mov     [[TMP:x.*]], xzr
; CHECK-O0: mov     x21, [[TMP]]
; CHECK-O0: bl      _foo2
; CHECK-O0: str     x21, [s[[STK:.*]]]
; CHECK-O0: ldr     x0, [s[[STK]]]

; CHECK-APPLE-LABEL: testAssign
; CHECK-APPLE: mov      x21, xzr
; CHECK-APPLE: bl      _foo2
; CHECK-APPLE: mov      x0, x21

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
