; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

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
  store i32 %foo, i32* %foo.addr, align 4
  %0 = load i32* %foo.addr, align 4
  store i32 %0, i32* @t, align 4
  ret void
}

define void @test() nounwind {
entry:
; 16: 	.frame	$16,24,$ra
; 16: 	save 	$ra, $s0, $s1, 24
; 16: 	move	$16, $sp
; 16:	move	${{[0-9]+}}, $sp
; 16:	subu	$[[REGISTER:[0-9]+]], ${{[0-9]+}}, ${{[0-9]+}}
; 16:	move	$sp, $[[REGISTER]]
  %sssi = alloca i32, align 4
  %ip = alloca i32*, align 4
  %sssj = alloca i32, align 4
  %0 = load i32* @iiii, align 4
  store i32 %0, i32* %sssi, align 4
  %1 = load i32* @kkkk, align 4
  %mul = mul nsw i32 %1, 100
  %2 = alloca i8, i32 %mul
  %3 = bitcast i8* %2 to i32*
  store i32* %3, i32** %ip, align 4
  %4 = load i32* @jjjj, align 4
  store i32 %4, i32* %sssj, align 4
  %5 = load i32* @jjjj, align 4
  %6 = load i32* @iiii, align 4
  %7 = load i32** %ip, align 4
  %arrayidx = getelementptr inbounds i32* %7, i32 %6
  store i32 %5, i32* %arrayidx, align 4
  %8 = load i32* @kkkk, align 4
  %9 = load i32* @jjjj, align 4
  %10 = load i32** %ip, align 4
  %arrayidx1 = getelementptr inbounds i32* %10, i32 %9
  store i32 %8, i32* %arrayidx1, align 4
  %11 = load i32* @iiii, align 4
  %12 = load i32* @kkkk, align 4
  %13 = load i32** %ip, align 4
  %arrayidx2 = getelementptr inbounds i32* %13, i32 %12
  store i32 %11, i32* %arrayidx2, align 4
  %14 = load i32** %ip, align 4
  %arrayidx3 = getelementptr inbounds i32* %14, i32 25
  %15 = load i32* %arrayidx3, align 4
  store i32 %15, i32* @riii, align 4
  %16 = load i32** %ip, align 4
  %arrayidx4 = getelementptr inbounds i32* %16, i32 35
  %17 = load i32* %arrayidx4, align 4
  store i32 %17, i32* @rjjj, align 4
  %18 = load i32** %ip, align 4
  %arrayidx5 = getelementptr inbounds i32* %18, i32 100
  %19 = load i32* %arrayidx5, align 4
  store i32 %19, i32* @rkkk, align 4
  %20 = load i32* @t, align 4
  %21 = load i32** %ip, align 4
  %arrayidx6 = getelementptr inbounds i32* %21, i32 %20
  %22 = load i32* %arrayidx6, align 4
; 16: 	save	16
  call void @temp(i32 %22)
; 16: 	restore	16
  ret void
}
