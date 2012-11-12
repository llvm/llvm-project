; RUN: opt -instcombine -S < %s | FileCheck %s

declare void @llvm.init.trampoline(i8*, i8*, i8*)
declare i8* @llvm.adjust.trampoline(i8*)
declare i32 @f(i8 * nest, i32)

; Most common case
define i32 @test0(i32 %n) {
  %alloca = alloca [10 x i8], align 16
  %gep = getelementptr [10 x i8]* %alloca, i32 0, i32 0
  call void @llvm.init.trampoline(i8* %gep, i8* bitcast (i32 (i8*, i32)* @f to i8*),
                                  i8* null)
  %tramp = call i8* @llvm.adjust.trampoline(i8* %gep)
  %function = bitcast i8* %tramp to i32(i32)*
  %ret = call i32 %function(i32 %n)
  ret i32 %ret

; CHECK: define i32 @test0(i32 %n) {
; CHECK: %ret = call i32 @f(i8* nest null, i32 %n)
}

define i32 @test1(i32 %n, i8* %trampmem) {
  call void @llvm.init.trampoline(i8* %trampmem,
                                  i8* bitcast (i32 (i8*, i32)* @f to i8*),
                                  i8* null)
  %tramp = call i8* @llvm.adjust.trampoline(i8* %trampmem)
  %function = bitcast i8* %tramp to i32(i32)*
  %ret = call i32 %function(i32 %n)
  ret i32 %ret
; CHECK: define i32 @test1(i32 %n, i8* %trampmem) {
; CHECK: %ret = call i32 @f(i8* nest null, i32 %n)
}

define i32 @test2(i32 %n, i8* %trampmem) {
  %tramp = call i8* @llvm.adjust.trampoline(i8* %trampmem)
  %functiona = bitcast i8* %tramp to i32(i32)*
  %ret = call i32 %functiona(i32 %n)
  ret i32 %ret
; CHECK: define i32 @test2(i32 %n, i8* %trampmem) {
; CHECK: %ret = call i32 %functiona(i32 %n)
}

define i32 @test3(i32 %n, i8* %trampmem) {
  call void @llvm.init.trampoline(i8* %trampmem,
                                  i8* bitcast (i32 (i8*, i32)* @f to i8*),
                                  i8* null)

; CHECK: define i32 @test3(i32 %n, i8* %trampmem) {
; CHECK: %ret0 = call i32 @f(i8* nest null, i32 %n)
  %tramp0 = call i8* @llvm.adjust.trampoline(i8* %trampmem)
  %function0 = bitcast i8* %tramp0 to i32(i32)*
  %ret0 = call i32 %function0(i32 %n)

  ;; Not optimized since previous call could be writing.
  %tramp1 = call i8* @llvm.adjust.trampoline(i8* %trampmem)
  %function1 = bitcast i8* %tramp1 to i32(i32)*
  %ret1 = call i32 %function1(i32 %n)
; CHECK: %ret1 = call i32 %function1(i32 %n)

  ret i32 %ret1
}

define i32 @test4(i32 %n) {
  %alloca = alloca [10 x i8], align 16
  %gep = getelementptr [10 x i8]* %alloca, i32 0, i32 0
  call void @llvm.init.trampoline(i8* %gep, i8* bitcast (i32 (i8*, i32)* @f to i8*),
                                  i8* null)

  %tramp0 = call i8* @llvm.adjust.trampoline(i8* %gep)
  %function0 = bitcast i8* %tramp0 to i32(i32)*
  %ret0 = call i32 %function0(i32 %n)

  %tramp1 = call i8* @llvm.adjust.trampoline(i8* %gep)
  %function1 = bitcast i8* %tramp0 to i32(i32)*
  %ret1 = call i32 %function1(i32 %n)

  %tramp2 = call i8* @llvm.adjust.trampoline(i8* %gep)
  %function2 = bitcast i8* %tramp2 to i32(i32)*
  %ret2 = call i32 %function2(i32 %n)

  ret i32 %ret2

; CHECK: define i32 @test4(i32 %n) {
; CHECK: %ret0 = call i32 @f(i8* nest null, i32 %n)
; CHECK: %ret1 = call i32 @f(i8* nest null, i32 %n)
; CHECK: %ret2 = call i32 @f(i8* nest null, i32 %n)
}
