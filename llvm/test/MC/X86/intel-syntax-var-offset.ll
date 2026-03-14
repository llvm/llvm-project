;RUN: llc -mtriple=x86_64-unknown-unknown -filetype=asm -x86-asm-syntax=intel < %s | FileCheck %s --check-prefix=CHECK
;PR34617

;Compile it with: "clang -O1 -emit-llvm"
;char X[4];
;volatile char* PX;
;char Y[4];
;volatile char* PY;
;char Z[4];
;volatile char* PZ;
;char* test057(long long x) {
;        asm ("movq %1, %%rax;"
;             "movq %%rax, %0;"
;             "pushq $Y;"
;             "popq %%rcx;"
;             "movq %%rcx, PY;"
;             "movq $X, %%rdx;"
;             "movq %%rdx, PX;"
;             :"=r"(PZ)
;             :"p"(Z)
;             :"%rax", "%rcx", "%rdx"
;             );
;    return (char*)PZ;
;}

; CHECK:	mov	rax, offset Z
; CHECK:	push	offset Y
; CHECK:	pop	rcx
; CHECK:	mov	qword ptr [PY], rcx
; CHECK:	mov	rdx, offset X
; CHECK:	mov	qword ptr [PX], rdx

@PZ = common dso_local global ptr null, align 8
@Z = common dso_local global [4 x i8] zeroinitializer, align 1
@X = common dso_local global [4 x i8] zeroinitializer, align 1
@PX = common dso_local global ptr null, align 8
@Y = common dso_local global [4 x i8] zeroinitializer, align 1
@PY = common dso_local global ptr null, align 8

define dso_local ptr @test057(i64 %x) {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, ptr %x.addr, align 8
  %0 = call ptr asm "movq $1, %rax;movq %rax, $0;pushq $$Y;popq %rcx;movq %rcx, PY;movq $$X, %rdx;movq %rdx, PX;", "=r,im,~{rax},~{rcx},~{rdx},~{dirflag},~{fpsr},~{flags}"(ptr @Z)
  store ptr %0, ptr @PZ, align 8
  %1 = load ptr, ptr @PZ, align 8
  ret ptr %1
}

