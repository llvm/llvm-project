; RUN: llc -mtriple=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=16

@i = global i32 25, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define void @p(ptr %i) nounwind {
entry:
  ret void
}


define void @foo() nounwind {
entry:
  %y = alloca [512 x i32], align 4
  %x = alloca i32, align 8
  %zz = alloca i32, align 4
  %z = alloca i32, align 4
  %0 = load i32, ptr @i, align 4
  %arrayidx = getelementptr inbounds [512 x i32], ptr %y, i32 0, i32 10
  store i32 %0, ptr %arrayidx, align 4
  %1 = load i32, ptr @i, align 4
  store i32 %1, ptr %x, align 8
  call void @p(ptr %x)
  %arrayidx1 = getelementptr inbounds [512 x i32], ptr %y, i32 0, i32 10
  call void @p(ptr %arrayidx1)
  ret void
}
; 16:	save	$ra, 2040
; 16:	addiu	$sp, -40 # 16 bit inst
; 16:	addiu	$sp, 40 # 16 bit inst
; 16:	restore	$ra, 2040
