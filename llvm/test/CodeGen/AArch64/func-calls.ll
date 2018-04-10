; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-neon | FileCheck --check-prefix=CHECK-NONEON %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64_be-none-linux-gnu | FileCheck --check-prefix=CHECK-BE %s

%myStruct = type { i64 , i8, i32 }

@var8 = global i8 0
@var8_2 = global i8 0
@var32 = global i32 0
@var64 = global i64 0
@var128 = global i128 0
@varfloat = global float 0.0
@varfloat_2 = global float 0.0
@vardouble = global double 0.0
@varstruct = global %myStruct zeroinitializer
@varsmallstruct = global [2 x i64] zeroinitializer

declare void @take_i8s(i8 %val1, i8 %val2)
declare void @take_floats(float %val1, float %val2)

define void @simple_args() {
; CHECK-LABEL: simple_args:
  %char1 = load i8, i8* @var8
  %char2 = load i8, i8* @var8_2
  call void @take_i8s(i8 %char1, i8 %char2)
; CHECK-DAG: ldrb w0, [{{x[0-9]+}}, {{#?}}:lo12:var8]
; CHECK-DAG: ldrb w1, [{{x[0-9]+}}, {{#?}}:lo12:var8_2]
; CHECK: bl take_i8s

  %float1 = load float, float* @varfloat
  %float2 = load float, float* @varfloat_2
  call void @take_floats(float %float1, float %float2)
; CHECK-DAG: ldr s1, [{{x[0-9]+}}, {{#?}}:lo12:varfloat_2]
; CHECK-DAG: ldr s0, [{{x[0-9]+}}, {{#?}}:lo12:varfloat]
; CHECK: bl take_floats
; CHECK-NOFP-NOT: ldr s1,
; CHECK-NOFP-NOT: ldr s0,

  ret void
}

declare i32 @return_int()
declare double @return_double()
declare [2 x i64] @return_smallstruct()
declare void @return_large_struct(%myStruct* sret %retval)

define void @simple_rets() {
; CHECK-LABEL: simple_rets:

  %int = call i32 @return_int()
  store i32 %int, i32* @var32
; CHECK: bl return_int
; CHECK: str w0, [{{x[0-9]+}}, {{#?}}:lo12:var32]

  %dbl = call double @return_double()
  store double %dbl, double* @vardouble
; CHECK: bl return_double
; CHECK: str d0, [{{x[0-9]+}}, {{#?}}:lo12:vardouble]
; CHECK-NOFP-NOT: str d0,

  %arr = call [2 x i64] @return_smallstruct()
  store [2 x i64] %arr, [2 x i64]* @varsmallstruct
; CHECK: bl return_smallstruct
; CHECK: add x[[VARSMALLSTRUCT:[0-9]+]], {{x[0-9]+}}, :lo12:varsmallstruct
; CHECK: stp x0, x1, [x[[VARSMALLSTRUCT]]]

  call void @return_large_struct(%myStruct* sret @varstruct)
; CHECK: add x8, {{x[0-9]+}}, {{#?}}:lo12:varstruct
; CHECK: bl return_large_struct

  ret void
}


declare i32 @struct_on_stack(i8 %var0, i16 %var1, i32 %var2, i64 %var3, i128 %var45,
                             i32* %var6, %myStruct* byval %struct, i32 %stacked,
                             double %notstacked)
declare void @stacked_fpu(float %var0, double %var1, float %var2, float %var3,
                          float %var4, float %var5, float %var6, float %var7,
                          float %var8)

define void @check_stack_args() {
; CHECK-LABEL: check_stack_args:
  call i32 @struct_on_stack(i8 0, i16 12, i32 42, i64 99, i128 1,
                            i32* @var32, %myStruct* byval @varstruct,
                            i32 999, double 1.0)
  ; Want to check that the final double is passed in registers and
  ; that varstruct is passed on the stack. Rather dependent on how a
  ; memcpy gets created, but the following works for now.

; CHECK-DAG: str {{q[0-9]+}}, [sp]
; CHECK-DAG: fmov d[[FINAL_DOUBLE:[0-9]+]], #1.0
; CHECK: mov v0.16b, v[[FINAL_DOUBLE]].16b

; CHECK-NONEON-DAG: str {{q[0-9]+}}, [sp]
; CHECK-NONEON-DAG: fmov d[[FINAL_DOUBLE:[0-9]+]], #1.0
; CHECK-NONEON: fmov d0, d[[FINAL_DOUBLE]]

; CHECK: bl struct_on_stack
; CHECK-NOFP-NOT: fmov

  call void @stacked_fpu(float -1.0, double 1.0, float 4.0, float 2.0,
                         float -2.0, float -8.0, float 16.0, float 1.0,
                         float 64.0)

; CHECK:  mov [[SIXTY_FOUR:w[0-9]+]], #1115684864
; CHECK: str [[SIXTY_FOUR]], [sp]

; CHECK-NONEON:  mov [[SIXTY_FOUR:w[0-9]+]], #1115684864
; CHECK-NONEON: str [[SIXTY_FOUR]], [sp]

; CHECK: bl stacked_fpu
  ret void
}


declare void @check_i128_stackalign(i32 %val0, i32 %val1, i32 %val2, i32 %val3,
                                    i32 %val4, i32 %val5, i32 %val6, i32 %val7,
                                    i32 %stack1, i128 %stack2)

declare void @check_i128_regalign(i32 %val0, i128 %val1)


define void @check_i128_align() {
; CHECK-LABEL: check_i128_align:
  %val = load i128, i128* @var128
  call void @check_i128_stackalign(i32 0, i32 1, i32 2, i32 3,
                                   i32 4, i32 5, i32 6, i32 7,
                                   i32 42, i128 %val)
; CHECK: add x[[VAR128:[0-9]+]], {{x[0-9]+}}, :lo12:var128
; CHECK: ldp [[I128LO:x[0-9]+]], [[I128HI:x[0-9]+]], [x[[VAR128]]]
; CHECK: stp [[I128HI]], {{x[0-9]+}}, [sp, #24]

; CHECK-NONEON: add x[[VAR128:[0-9]+]], {{x[0-9]+}}, :lo12:var128
; CHECK-NONEON: ldp [[I128LO:x[0-9]+]], [[I128HI:x[0-9]+]], [x[[VAR128]]]
; CHECK-NONEON: stp [[I128HI]], {{x[0-9]+}}, [sp, #24]
; CHECK: bl check_i128_stackalign

  call void @check_i128_regalign(i32 0, i128 42)
; CHECK-NOT: mov x1
; CHECK-LE: mov x2, #{{0x2a|42}}
; CHECK-LE: mov x3, xzr
; CHECK-BE: mov {{x|w}}3, #{{0x2a|42}}
; CHECK-BE: mov x2, xzr
; CHECK: bl check_i128_regalign

  ret void
}

@fptr = global void()* null

define void @check_indirect_call() {
; CHECK-LABEL: check_indirect_call:
  %func = load void()*, void()** @fptr
  call void %func()
; CHECK: ldr [[FPTR:x[0-9]+]], [{{x[0-9]+}}, {{#?}}:lo12:fptr]
; CHECK: blr [[FPTR]]

  ret void
}
