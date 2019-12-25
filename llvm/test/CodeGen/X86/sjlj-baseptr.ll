; RUN: llc < %s -mtriple=i386-pc-linux -mcpu=corei7 -relocation-model=static | FileCheck --check-prefix=X86 %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7 -relocation-model=static | FileCheck --check-prefix=X64 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%Foo = type { [125 x i8] }

declare i32 @llvm.eh.sjlj.setjmp(i8*) nounwind

declare void @whatever(i64, %Foo*, i8**, i8*, i8*, i32)  #0

attributes #0 = { nounwind uwtable "frame-pointer"="all" }

define i32 @test1(i64 %n, %Foo* byval nocapture readnone align 8 %f) #0 {
entry:
  %buf = alloca [5 x i8*], align 16
  %p = alloca i8*, align 8
  %q = alloca i8, align 64
  %r = bitcast [5 x i8*]* %buf to i8*
  %s = alloca i8, i64 %n, align 1
  store i8* %s, i8** %p, align 8
  %t = call i32 @llvm.eh.sjlj.setjmp(i8* %s)
  call void @whatever(i64 %n, %Foo* %f, i8** %p, i8* %q, i8* %s, i32 %t) #1
  ret i32 0
; X86: movl    %esp, %esi
; X86: movl    %esp, -16(%ebp)
; X86: {{.LBB.*:}}
; X86: movl    -16(%ebp), %esi
; X86: {{.LBB.*:}}
; X64: movq    %rsp, %rbx
; X64: movq    %rsp, -48(%rbp)
; X64: {{.LBB.*:}}
; X64: movq    -48(%rbp), %rbx
; X64: {{.LBB.*:}}
}


