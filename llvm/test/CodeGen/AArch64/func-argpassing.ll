; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -disable-post-ra | FileCheck --check-prefix=CHECK %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 -disable-post-ra | FileCheck --check-prefix=CHECK-NOFP %s

%myStruct = type { i64 , i8, i32 }

@var8 = global i8 0
@var32 = global i32 0
@var64 = global i64 0
@var128 = global i128 0
@varfloat = global float 0.0
@vardouble = global double 0.0
@varstruct = global %myStruct zeroinitializer

define void @take_i8s(i8 %val1, i8 %val2) {
; CHECK-LABEL: take_i8s:
    store i8 %val2, i8* @var8
    ; Not using w1 may be technically allowed, but it would indicate a
    ; problem in itself.
;  CHECK: strb w1, [{{x[0-9]+}}, {{#?}}:lo12:var8]
    ret void
}

define void @add_floats(float %val1, float %val2) {
; CHECK-LABEL: add_floats:
    %newval = fadd float %val1, %val2
; CHECK: fadd [[ADDRES:s[0-9]+]], s0, s1
; CHECK-NOFP-NOT: fadd
    store float %newval, float* @varfloat
; CHECK: str [[ADDRES]], [{{x[0-9]+}}, {{#?}}:lo12:varfloat]
    ret void
}

; byval pointers should be allocated to the stack and copied as if
; with memcpy.
define void @take_struct(%myStruct* byval %structval) {
; CHECK-LABEL: take_struct:
    %addr0 = getelementptr %myStruct, %myStruct* %structval, i64 0, i32 2
    %addr1 = getelementptr %myStruct, %myStruct* %structval, i64 0, i32 0

    %val0 = load volatile i32, i32* %addr0
    ; Some weird move means x0 is used for one access
; CHECK: ldr [[REG32:w[0-9]+]], [{{x[0-9]+|sp}}, #12]
    store volatile i32 %val0, i32* @var32
; CHECK: str [[REG32]], [{{x[0-9]+}}, {{#?}}:lo12:var32]

    %val1 = load volatile i64, i64* %addr1
; CHECK: ldr [[REG64:x[0-9]+]], [{{x[0-9]+|sp}}]
    store volatile i64 %val1, i64* @var64
; CHECK: str [[REG64]], [{{x[0-9]+}}, {{#?}}:lo12:var64]

    ret void
}

; %structval should be at sp + 16
define void @check_byval_align(i32* byval %ignore, %myStruct* byval align 16 %structval) {
; CHECK-LABEL: check_byval_align:

    %addr0 = getelementptr %myStruct, %myStruct* %structval, i64 0, i32 2
    %addr1 = getelementptr %myStruct, %myStruct* %structval, i64 0, i32 0

    %val0 = load volatile i32, i32* %addr0
    ; Some weird move means x0 is used for one access
; CHECK: ldr [[REG32:w[0-9]+]], [sp, #28]
    store i32 %val0, i32* @var32
; CHECK: str [[REG32]], [{{x[0-9]+}}, {{#?}}:lo12:var32]

    %val1 = load volatile i64, i64* %addr1
; CHECK: ldr [[REG64:x[0-9]+]], [sp, #16]
    store i64 %val1, i64* @var64
; CHECK: str [[REG64]], [{{x[0-9]+}}, {{#?}}:lo12:var64]

    ret void
}

define i32 @return_int() {
; CHECK-LABEL: return_int:
    %val = load i32, i32* @var32
    ret i32 %val
; CHECK: ldr w0, [{{x[0-9]+}}, {{#?}}:lo12:var32]
    ; Make sure epilogue follows
; CHECK-NEXT: ret
}

define double @return_double() {
; CHECK-LABEL: return_double:
    ret double 3.14
; CHECK: ldr d0, [{{x[0-9]+}}, {{#?}}:lo12:.LCPI
; CHECK-NOFP-NOT: ldr d0,
}

; This is the kind of IR clang will produce for returning a struct
; small enough to go into registers. Not all that pretty, but it
; works.
define [2 x i64] @return_struct() {
; CHECK-LABEL: return_struct:
    %addr = bitcast %myStruct* @varstruct to [2 x i64]*
    %val = load [2 x i64], [2 x i64]* %addr
    ret [2 x i64] %val
; CHECK: add x[[VARSTRUCT:[0-9]+]], {{x[0-9]+}}, :lo12:varstruct
; CHECK: ldp x0, x1, [x[[VARSTRUCT]]]
    ; Make sure epilogue immediately follows
; CHECK-NEXT: ret
}

; Large structs are passed by reference (storage allocated by caller
; to preserve value semantics) in x8. Strictly this only applies to
; structs larger than 16 bytes, but C semantics can still be provided
; if LLVM does it to %myStruct too. So this is the simplest check
define void @return_large_struct(%myStruct* sret %retval) {
; CHECK-LABEL: return_large_struct:
    %addr0 = getelementptr %myStruct, %myStruct* %retval, i64 0, i32 0
    %addr1 = getelementptr %myStruct, %myStruct* %retval, i64 0, i32 1
    %addr2 = getelementptr %myStruct, %myStruct* %retval, i64 0, i32 2

    store i64 42, i64* %addr0
    store i8 2, i8* %addr1
    store i32 9, i32* %addr2
; CHECK: str {{x[0-9]+}}, [x8]
; CHECK: strb {{w[0-9]+}}, [x8, #8]
; CHECK: str {{w[0-9]+}}, [x8, #12]

    ret void
}

; This struct is just too far along to go into registers: (only x7 is
; available, but it needs two). Also make sure that %stacked doesn't
; sneak into x7 behind.
define i32 @struct_on_stack(i8 %var0, i16 %var1, i32 %var2, i64 %var3, i128 %var45,
                          i32* %var6, %myStruct* byval %struct, i32* byval %stacked,
                          double %notstacked) {
; CHECK-LABEL: struct_on_stack:
    %addr = getelementptr %myStruct, %myStruct* %struct, i64 0, i32 0
    %val64 = load volatile i64, i64* %addr
    store volatile i64 %val64, i64* @var64
    ; Currently nothing on local stack, so struct should be at sp
; CHECK: ldr [[VAL64:x[0-9]+]], [sp]
; CHECK: str [[VAL64]], [{{x[0-9]+}}, {{#?}}:lo12:var64]

    store volatile double %notstacked, double* @vardouble
; CHECK-NOT: ldr d0
; CHECK: str d0, [{{x[0-9]+}}, {{#?}}:lo12:vardouble
; CHECK-NOFP-NOT: str d0,

    %retval = load volatile i32, i32* %stacked
    ret i32 %retval
; CHECK-LE: ldr w0, [sp, #16]
}

define void @stacked_fpu(float %var0, double %var1, float %var2, float %var3,
                         float %var4, float %var5, float %var6, float %var7,
                         float %var8) {
; CHECK-LABEL: stacked_fpu:
    store float %var8, float* @varfloat
    ; Beware as above: the offset would be different on big-endian
    ; machines if the first ldr were changed to use s-registers.
; CHECK: ldr {{[ds]}}[[VALFLOAT:[0-9]+]], [sp]
; CHECK: str s[[VALFLOAT]], [{{x[0-9]+}}, {{#?}}:lo12:varfloat]

    ret void
}

; 128-bit integer types should be passed in xEVEN, xODD rather than
; the reverse. In this case x2 and x3. Nothing should use x1.
define i64 @check_i128_regalign(i32 %val0, i128 %val1, i64 %val2) {
; CHECK-LABEL: check_i128_regalign
    store i128 %val1, i128* @var128
; CHECK: add x[[VAR128:[0-9]+]], {{x[0-9]+}}, :lo12:var128
; CHECK-DAG: stp x2, x3, [x[[VAR128]]]

    ret i64 %val2
; CHECK: mov x0, x4
}

define void @check_i128_stackalign(i32 %val0, i32 %val1, i32 %val2, i32 %val3,
                                   i32 %val4, i32 %val5, i32 %val6, i32 %val7,
                                   i32 %stack1, i128 %stack2) {
; CHECK-LABEL: check_i128_stackalign
    store i128 %stack2, i128* @var128
    ; Nothing local on stack in current codegen, so first stack is 16 away
; CHECK-LE: add     x[[REG:[0-9]+]], sp, #16
; CHECK-LE: ldr {{x[0-9]+}}, [x[[REG]], #8]

    ; Important point is that we address sp+24 for second dword

; CHECK: ldp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
    ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)

define i32 @test_extern() {
; CHECK-LABEL: test_extern:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* undef, i32 undef, i32 4, i1 0)
; CHECK: bl memcpy
  ret i32 0
}


; A sub-i32 stack argument must be loaded on big endian with ldr{h,b}, not just
; implicitly extended to a 32-bit load.
define i16 @stacked_i16(i32 %val0, i32 %val1, i32 %val2, i32 %val3,
                        i32 %val4, i32 %val5, i32 %val6, i32 %val7,
                        i16 %stack1) {
; CHECK-LABEL: stacked_i16
  ret i16 %stack1
}
