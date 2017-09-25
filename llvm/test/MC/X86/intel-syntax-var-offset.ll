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

@PZ = common global i8* null, align 8
@Z = common global [4 x i8] zeroinitializer, align 1
@X = common global [4 x i8] zeroinitializer, align 1
@PX = common global i8* null, align 8
@Y = common global [4 x i8] zeroinitializer, align 1
@PY = common global i8* null, align 8

define i8* @test057(i64 %x) {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  %0 = call i8* asm "movq $1, %rax;movq %rax, $0;pushq $$Y;popq %rcx;movq %rcx, PY;movq $$X, %rdx;movq %rdx, PX;", "=r,im,~{rax},~{rcx},~{rdx},~{dirflag},~{fpsr},~{flags}"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @Z, i32 0, i32 0))
  store i8* %0, i8** @PZ, align 8
  %1 = load i8*, i8** @PZ, align 8
  ret i8* %1
}

