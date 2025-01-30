; RUN: llc -mtriple=mipsel-elf -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=16

@t = global i32 10, align 4
@f = global i32 199, align 4
@a = global i32 2, align 4
@b = global i32 1000, align 4
@c = global i32 2, align 4
@z1 = common global i32 0, align 4
@z2 = common global i32 0, align 4
@z3 = common global i32 0, align 4
@z4 = common global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define void @calc_selltk() nounwind "target-cpu"="mips16" "target-features"="+mips16,+o32" {
entry:
  %0 = load i32, ptr @a, align 4
  %cmp = icmp slt i32 %0, 1000
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %1 = load i32, ptr @t, align 4
  br label %cond.end

cond.false:                                       ; preds = %entry
  %2 = load i32, ptr @f, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %1, %cond.true ], [ %2, %cond.false ]
  store i32 %cond, ptr @z1, align 4
  %3 = load i32, ptr @b, align 4
  %cmp1 = icmp slt i32 %3, 2
  br i1 %cmp1, label %cond.true2, label %cond.false3

cond.true2:                                       ; preds = %cond.end
  %4 = load i32, ptr @f, align 4
  br label %cond.end4

cond.false3:                                      ; preds = %cond.end
  %5 = load i32, ptr @t, align 4
  br label %cond.end4

cond.end4:                                        ; preds = %cond.false3, %cond.true2
  %cond5 = phi i32 [ %4, %cond.true2 ], [ %5, %cond.false3 ]
  store i32 %cond5, ptr @z2, align 4
  %6 = load i32, ptr @c, align 4
  %cmp6 = icmp sgt i32 %6, 2
  br i1 %cmp6, label %cond.true7, label %cond.false8

cond.true7:                                       ; preds = %cond.end4
  %7 = load i32, ptr @f, align 4
  br label %cond.end9

cond.false8:                                      ; preds = %cond.end4
  %8 = load i32, ptr @t, align 4
  br label %cond.end9

cond.end9:                                        ; preds = %cond.false8, %cond.true7
  %cond10 = phi i32 [ %7, %cond.true7 ], [ %8, %cond.false8 ]
  store i32 %cond10, ptr @z3, align 4
  %9 = load i32, ptr @a, align 4
  %cmp11 = icmp sgt i32 %9, 2
  br i1 %cmp11, label %cond.true12, label %cond.false13

cond.true12:                                      ; preds = %cond.end9
  %10 = load i32, ptr @f, align 4
  br label %cond.end14

cond.false13:                                     ; preds = %cond.end9
  %11 = load i32, ptr @t, align 4
  br label %cond.end14

cond.end14:                                       ; preds = %cond.false13, %cond.true12
  %cond15 = phi i32 [ %10, %cond.true12 ], [ %11, %cond.false13 ]
  store i32 %cond15, ptr @z4, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="mips16" "target-features"="+mips16,+o32" }
attributes #1 = { "target-cpu"="mips16" "target-features"="+mips16,+o32" }

; 16:	slt	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

; 16:	slt	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

; 16:	slti	${{[0-9]+}}, 3 	# 16 bit inst
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

; 16:	slti	${{[0-9]+}}, 3 	# 16 bit inst
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

