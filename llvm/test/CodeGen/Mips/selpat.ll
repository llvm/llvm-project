; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@t = global i32 10, align 4
@f = global i32 199, align 4
@a = global i32 1, align 4
@b = global i32 10, align 4
@c = global i32 1, align 4
@z1 = common global i32 0, align 4
@z2 = common global i32 0, align 4
@z3 = common global i32 0, align 4
@z4 = common global i32 0, align 4

define void @calc_seleq() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp eq i32 %0, %1
  %2 = load i32* @f, align 4
  %3 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  store i32 %cond, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp eq i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %3, i32 %2
  store i32 %cond10, i32* @z3, align 4
  store i32 %cond10, i32* @z4, align 4
  ret void
}


define void @calc_seleqk() nounwind {
entry:
  %0 = load i32* @a, align 4
  %cmp = icmp eq i32 %0, 1
  %1 = load i32* @t, align 4
  %2 = load i32* @f, align 4
  %cond = select i1 %cmp, i32 %1, i32 %2
  store i32 %cond, i32* @z1, align 4
; 16:	cmpi	${{[0-9]+}}, 1
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp eq i32 %0, 10
  %cond5 = select i1 %cmp1, i32 %2, i32 %1
  store i32 %cond5, i32* @z2, align 4
  %3 = load i32* @b, align 4
  %cmp6 = icmp eq i32 %3, 3
  %cond10 = select i1 %cmp6, i32 %2, i32 %1
  store i32 %cond10, i32* @z3, align 4
; 16:	cmpi	${{[0-9]+}}, 10
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp11 = icmp eq i32 %3, 10
  %cond15 = select i1 %cmp11, i32 %1, i32 %2
  store i32 %cond15, i32* @z4, align 4
  ret void
}

define void @calc_seleqz() nounwind {
entry:
  %0 = load i32* @a, align 4
  %cmp = icmp eq i32 %0, 0
  %1 = load i32* @t, align 4
  %2 = load i32* @f, align 4
  %cond = select i1 %cmp, i32 %1, i32 %2
  store i32 %cond, i32* @z1, align 4
; 16:	beqz	${{[0-9]+}}, .+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %3 = load i32* @b, align 4
  %cmp1 = icmp eq i32 %3, 0
  %cond5 = select i1 %cmp1, i32 %2, i32 %1
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp eq i32 %4, 0
  %cond10 = select i1 %cmp6, i32 %1, i32 %2
  store i32 %cond10, i32* @z3, align 4
  store i32 %cond, i32* @z4, align 4
  ret void
}

define void @calc_selge() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp sge i32 %0, %1
  %2 = load i32* @f, align 4
  %3 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	slt	${{[0-9]+}}, ${{[0-9]+}}
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp sge i32 %1, %0
  %cond5 = select i1 %cmp1, i32 %3, i32 %2
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp sge i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %3, i32 %2
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp sge i32 %0, %4
  %cond15 = select i1 %cmp11, i32 %3, i32 %2
  store i32 %cond15, i32* @z4, align 4
  ret void
}

define i32 @calc_selgt() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp sgt i32 %0, %1
; 16:	slt	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %2 = load i32* @f, align 4
  %3 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
  %cmp1 = icmp sgt i32 %1, %0
  %cond5 = select i1 %cmp1, i32 %3, i32 %2
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp sgt i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %2, i32 %3
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp sgt i32 %0, %4
  %cond15 = select i1 %cmp11, i32 %2, i32 %3
  store i32 %cond15, i32* @z4, align 4
  ret i32 undef
}

define void @calc_selle() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp sle i32 %0, %1
  %2 = load i32* @t, align 4
  %3 = load i32* @f, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	slt	${{[0-9]+}}, ${{[0-9]+}}
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp sle i32 %1, %0
  %cond5 = select i1 %cmp1, i32 %3, i32 %2
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp sle i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %2, i32 %3
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp sle i32 %0, %4
  %cond15 = select i1 %cmp11, i32 %2, i32 %3
  store i32 %cond15, i32* @z4, align 4
  ret void
}

define void @calc_selltk() nounwind {
entry:
  %0 = load i32* @a, align 4
  %cmp = icmp slt i32 %0, 10
  %1 = load i32* @t, align 4
  %2 = load i32* @f, align 4
  %cond = select i1 %cmp, i32 %1, i32 %2
  store i32 %cond, i32* @z1, align 4
; 16:	slti	${{[0-9]+}}, {{[0-9]+}}
; 16:	btnez	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %3 = load i32* @b, align 4
  %cmp1 = icmp slt i32 %3, 2
  %cond5 = select i1 %cmp1, i32 %2, i32 %1
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp sgt i32 %4, 2
  %cond10 = select i1 %cmp6, i32 %2, i32 %1
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp sgt i32 %0, 2
  %cond15 = select i1 %cmp11, i32 %2, i32 %1
  store i32 %cond15, i32* @z4, align 4
  ret void
}


define void @calc_selne() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp ne i32 %0, %1
  %2 = load i32* @t, align 4
  %3 = load i32* @f, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  store i32 %cond, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp ne i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %3, i32 %2
  store i32 %cond10, i32* @z3, align 4
  store i32 %cond10, i32* @z4, align 4
  ret void
}

define void @calc_selnek() nounwind {
entry:
  %0 = load i32* @a, align 4
  %cmp = icmp ne i32 %0, 1
  %1 = load i32* @f, align 4
  %2 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %1, i32 %2
  store i32 %cond, i32* @z1, align 4
; 16:	cmpi	${{[0-9]+}}, 1
; 16:	btnez	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp ne i32 %0, 10
  %cond5 = select i1 %cmp1, i32 %2, i32 %1
  store i32 %cond5, i32* @z2, align 4
  %3 = load i32* @b, align 4
  %cmp6 = icmp ne i32 %3, 3
  %cond10 = select i1 %cmp6, i32 %2, i32 %1
  store i32 %cond10, i32* @z3, align 4
; 16:	cmpi	${{[0-9]+}}, 10
; 16:	btnez	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp11 = icmp ne i32 %3, 10
  %cond15 = select i1 %cmp11, i32 %1, i32 %2
  store i32 %cond15, i32* @z4, align 4
  ret void
}

define void @calc_selnez() nounwind {
entry:
  %0 = load i32* @a, align 4
  %cmp = icmp ne i32 %0, 0
  %1 = load i32* @f, align 4
  %2 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %1, i32 %2
  store i32 %cond, i32* @z1, align 4
; 16:	bnez	${{[0-9]+}}, .+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %3 = load i32* @b, align 4
  %cmp1 = icmp ne i32 %3, 0
  %cond5 = select i1 %cmp1, i32 %2, i32 %1
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp ne i32 %4, 0
  %cond10 = select i1 %cmp6, i32 %1, i32 %2
  store i32 %cond10, i32* @z3, align 4
  store i32 %cond, i32* @z4, align 4
  ret void
}

define void @calc_selnez2() nounwind {
entry:
  %0 = load i32* @a, align 4
  %tobool = icmp ne i32 %0, 0
  %1 = load i32* @f, align 4
  %2 = load i32* @t, align 4
  %cond = select i1 %tobool, i32 %1, i32 %2
  store i32 %cond, i32* @z1, align 4
; 16:	bnez	${{[0-9]+}}, .+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %3 = load i32* @b, align 4
  %tobool1 = icmp ne i32 %3, 0
  %cond5 = select i1 %tobool1, i32 %2, i32 %1
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %tobool6 = icmp ne i32 %4, 0
  %cond10 = select i1 %tobool6, i32 %1, i32 %2
  store i32 %cond10, i32* @z3, align 4
  store i32 %cond, i32* @z4, align 4
  ret void
}

define void @calc_seluge() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp uge i32 %0, %1
  %2 = load i32* @f, align 4
  %3 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	sltu	${{[0-9]+}}, ${{[0-9]+}}
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp uge i32 %1, %0
  %cond5 = select i1 %cmp1, i32 %3, i32 %2
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp uge i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %3, i32 %2
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp uge i32 %0, %4
  %cond15 = select i1 %cmp11, i32 %3, i32 %2
  store i32 %cond15, i32* @z4, align 4
  ret void
}

define void @calc_selugt() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp ugt i32 %0, %1
  %2 = load i32* @f, align 4
  %3 = load i32* @t, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	sltu	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp ugt i32 %1, %0
  %cond5 = select i1 %cmp1, i32 %3, i32 %2
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp ugt i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %2, i32 %3
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp ugt i32 %0, %4
  %cond15 = select i1 %cmp11, i32 %2, i32 %3
  store i32 %cond15, i32* @z4, align 4
  ret void
}

define void @calc_selule() nounwind {
entry:
  %0 = load i32* @a, align 4
  %1 = load i32* @b, align 4
  %cmp = icmp ule i32 %0, %1
  %2 = load i32* @t, align 4
  %3 = load i32* @f, align 4
  %cond = select i1 %cmp, i32 %2, i32 %3
  store i32 %cond, i32* @z1, align 4
; 16:	sltu	${{[0-9]+}}, ${{[0-9]+}}
; 16:	bteqz	.+4
; 16: 	move    ${{[0-9]+}}, ${{[0-9]+}}
  %cmp1 = icmp ule i32 %1, %0
  %cond5 = select i1 %cmp1, i32 %3, i32 %2
  store i32 %cond5, i32* @z2, align 4
  %4 = load i32* @c, align 4
  %cmp6 = icmp ule i32 %4, %0
  %cond10 = select i1 %cmp6, i32 %2, i32 %3
  store i32 %cond10, i32* @z3, align 4
  %cmp11 = icmp ule i32 %0, %4
  %cond15 = select i1 %cmp11, i32 %2, i32 %3
  store i32 %cond15, i32* @z4, align 4
  ret void
}
