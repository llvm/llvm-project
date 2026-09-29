; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@iiii = global i32 25, align 4
@jjjj = global i32 35, align 4
@kkkk = global i32 100, align 4
@t = global i32 25, align 4
@riii = common global i32 0, align 4
@rjjj = common global i32 0, align 4
@rkkk = common global i32 0, align 4

define void @temp(i32 %foo) nounwind {
entry:
  %foo.addr = alloca i32, align 4
  store i32 %foo, ptr %foo.addr, align 4
  %0 = load i32, ptr %foo.addr, align 4
  store i32 %0, ptr @t, align 4
  ret void
}

define void @test() nounwind {
entry:
; 16: 	.frame	$sp,8,$ra
; 16: 	save 	8 # 16 bit inst
; 16: 	move	$16, $sp
; 16:	move	${{[0-9]+}}, $sp
; 16:	subu	$[[REGISTER:[0-9]+]], ${{[0-9]+}}, ${{[0-9]+}}
; 16:	move	$sp, $[[REGISTER]]
  %sssi = alloca i32, align 4
  %ip = alloca ptr, align 4
  %sssj = alloca i32, align 4
  %0 = load i32, ptr @iiii, align 4
  store i32 %0, ptr %sssi, align 4
  %1 = load i32, ptr @kkkk, align 4
  %mul = mul nsw i32 %1, 100
  %2 = alloca i8, i32 %mul
  store ptr %2, ptr %ip, align 4
  %3 = load i32, ptr @jjjj, align 4
  store i32 %3, ptr %sssj, align 4
  %4 = load i32, ptr @jjjj, align 4
  %5 = load i32, ptr @iiii, align 4
  %6 = load ptr, ptr %ip, align 4
  %arrayidx = getelementptr inbounds i32, ptr %6, i32 %5
  store i32 %4, ptr %arrayidx, align 4
  %7 = load i32, ptr @kkkk, align 4
  %8 = load i32, ptr @jjjj, align 4
  %9 = load ptr, ptr %ip, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %9, i32 %8
  store i32 %7, ptr %arrayidx1, align 4
  %10 = load i32, ptr @iiii, align 4
  %11 = load i32, ptr @kkkk, align 4
  %12 = load ptr, ptr %ip, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %12, i32 %11
  store i32 %10, ptr %arrayidx2, align 4
  %13 = load ptr, ptr %ip, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %13, i32 25
  %14 = load i32, ptr %arrayidx3, align 4
  store i32 %14, ptr @riii, align 4
  %15 = load ptr, ptr %ip, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %15, i32 35
  %16 = load i32, ptr %arrayidx4, align 4
  store i32 %16, ptr @rjjj, align 4
  %17 = load ptr, ptr %ip, align 4
  %arrayidx5 = getelementptr inbounds i32, ptr %17, i32 100
  %18 = load i32, ptr %arrayidx5, align 4
  store i32 %18, ptr @rkkk, align 4
  %19 = load i32, ptr @t, align 4
  %20 = load ptr, ptr %ip, align 4
  %arrayidx6 = getelementptr inbounds i32, ptr %20, i32 %19
  %21 = load i32, ptr %arrayidx6, align 4
; 16: 	addiu $sp, -16
  call void @temp(i32 %21)
; 16: 	addiu $sp, 16
  ret void
}
