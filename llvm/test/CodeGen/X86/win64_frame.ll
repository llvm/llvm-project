; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s

define i32 @f1(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f1:
  ; CHECK:       movl    48(%rbp), %eax
  ret i32 %p5
}

define void @f2(i32 %p, ...) "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f2:
  ; CHECK:      .seh_stackalloc 8
  ; CHECK:      movq    %rsp, %rbp
  ; CHECK:      .seh_setframe 5, 0
  ; CHECK:      movq    %rdx, 32(%rbp)
  ; CHECK:      leaq    32(%rbp), %rax
  %ap = alloca i8, align 8
  call void @llvm.va_start(i8* %ap)
  ret void
}

define i8* @f3() "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f3:
  ; CHECK:      movq    %rsp, %rbp
  ; CHECK:      .seh_setframe 5, 0
  ; CHECK:      movq    8(%rbp), %rax
  %ra = call i8* @llvm.returnaddress(i32 0)
  ret i8* %ra
}

define i8* @f4() "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f4:
  ; CHECK:      pushq   %rbp
  ; CHECK:      .seh_pushreg 5
  ; CHECK:      subq    $304, %rsp
  ; CHECK:      .seh_stackalloc 304
  ; CHECK:      leaq    128(%rsp), %rbp
  ; CHECK:      .seh_setframe 5, 128
  ; CHECK:      .seh_endprologue
  ; CHECK:      movq    184(%rbp), %rax
  alloca [300 x i8]
  %ra = call i8* @llvm.returnaddress(i32 0)
  ret i8* %ra
}

declare void @external(i8*)

define void @f5() "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f5:
  ; CHECK:      subq    $336, %rsp
  ; CHECK:      .seh_stackalloc 336
  ; CHECK:      leaq    128(%rsp), %rbp
  ; CHECK:      .seh_setframe 5, 128
  ; CHECK:      leaq    -92(%rbp), %rcx
  ; CHECK:      callq   external
  %a = alloca [300 x i8]
  %gep = getelementptr [300 x i8], [300 x i8]* %a, i32 0, i32 0
  call void @external(i8* %gep)
  ret void
}

define void @f6(i32 %p, ...) "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f6:
  ; CHECK:      subq    $336, %rsp
  ; CHECK:      .seh_stackalloc 336
  ; CHECK:      leaq    128(%rsp), %rbp
  ; CHECK:      .seh_setframe 5, 128
  ; CHECK:      leaq    -92(%rbp), %rcx
  ; CHECK:      callq   external
  %a = alloca [300 x i8]
  %gep = getelementptr [300 x i8], [300 x i8]* %a, i32 0, i32 0
  call void @external(i8* %gep)
  ret void
}

define i32 @f7(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f7:
  ; CHECK:      pushq   %rbp
  ; CHECK:      .seh_pushreg 5
  ; CHECK:      subq    $304, %rsp
  ; CHECK:      .seh_stackalloc 304
  ; CHECK:      leaq    128(%rsp), %rbp
  ; CHECK:      .seh_setframe 5, 128
  ; CHECK:      andq    $-64, %rsp
  ; CHECK:      movl    224(%rbp), %eax
  ; CHECK:      leaq    176(%rbp), %rsp
  alloca [300 x i8], align 64
  ret i32 %e
}

define i32 @f8(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) "no-frame-pointer-elim"="true" {
  ; CHECK-LABEL: f8:
  ; CHECK:        subq    $352, %rsp
  ; CHECK:        .seh_stackalloc 352
  ; CHECK:        leaq    128(%rsp), %rbp
  ; CHECK:        .seh_setframe 5, 128

  %alloca = alloca [300 x i8], align 64
  ; CHECK:        andq    $-64, %rsp
  ; CHECK:        movq    %rsp, %rbx

  alloca i32, i32 %a
  ; CHECK:        movl    %ecx, %eax
  ; CHECK:        leaq    15(,%rax,4), %rcx
  ; CHECK:        movabsq $34359738352, %rax
  ; CHECK:        andq    %rcx, %rax
  ; CHECK:        callq   __chkstk
  ; CHECK:        subq    %rax, %rsp

  %gep = getelementptr [300 x i8], [300 x i8]* %alloca, i32 0, i32 0
  call void @external(i8* %gep)
  ; CHECK:        subq    $32, %rsp
  ; CHECK:        leaq    (%rbx), %rcx
  ; CHECK:        callq   external
  ; CHECK:        addq    $32, %rsp

  ret i32 %e
  ; CHECK:        movl    %esi, %eax
  ; CHECK:        leaq    224(%rbp), %rsp
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

declare void @llvm.va_start(i8*) nounwind
