; RUN: llc -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=16

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
  %0 = load i32, i32* @a, align 4
  %1 = load i32, i32* @b, align 4
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %2 = load i32, i32* @f, align 4
  br label %cond.end

cond.false:                                       ; preds = %entry
  %3 = load i32, i32* @t, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %2, %cond.true ], [ %3, %cond.false ]
  store i32 %cond, i32* @z1, align 4
  %4 = load i32, i32* @b, align 4
  %5 = load i32, i32* @a, align 4
  %cmp1 = icmp eq i32 %4, %5
  br i1 %cmp1, label %cond.true2, label %cond.false3

cond.true2:                                       ; preds = %cond.end
  %6 = load i32, i32* @f, align 4
  br label %cond.end4

cond.false3:                                      ; preds = %cond.end
  %7 = load i32, i32* @t, align 4
  br label %cond.end4

cond.end4:                                        ; preds = %cond.false3, %cond.true2
  %cond5 = phi i32 [ %6, %cond.true2 ], [ %7, %cond.false3 ]
  store i32 %cond5, i32* @z2, align 4
  %8 = load i32, i32* @c, align 4
  %9 = load i32, i32* @a, align 4
  %cmp6 = icmp eq i32 %8, %9
  br i1 %cmp6, label %cond.true7, label %cond.false8

cond.true7:                                       ; preds = %cond.end4
  %10 = load i32, i32* @t, align 4
  br label %cond.end9

cond.false8:                                      ; preds = %cond.end4
  %11 = load i32, i32* @f, align 4
  br label %cond.end9

cond.end9:                                        ; preds = %cond.false8, %cond.true7
  %cond10 = phi i32 [ %10, %cond.true7 ], [ %11, %cond.false8 ]
  store i32 %cond10, i32* @z3, align 4
  %12 = load i32, i32* @a, align 4
  %13 = load i32, i32* @c, align 4
  %cmp11 = icmp eq i32 %12, %13
  br i1 %cmp11, label %cond.true12, label %cond.false13

cond.true12:                                      ; preds = %cond.end9
  %14 = load i32, i32* @t, align 4
  br label %cond.end14

cond.false13:                                     ; preds = %cond.end9
  %15 = load i32, i32* @f, align 4
  br label %cond.end14

cond.end14:                                       ; preds = %cond.false13, %cond.true12
  %cond15 = phi i32 [ %14, %cond.true12 ], [ %15, %cond.false13 ]
  store i32 %cond15, i32* @z4, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="mips32" "target-features"="+o32,+mips32" }

; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$BB{{[0-9]+}}_{{[0-9]}}

