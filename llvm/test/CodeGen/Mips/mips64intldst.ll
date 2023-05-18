; RUN: llc  < %s -march=mips64el -mcpu=mips4 -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N64
; RUN: llc  < %s -march=mips64el -mcpu=mips4 -target-abi n32 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N32
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N64
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -target-abi n32 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N32

@c = common global i8 0, align 4
@s = common global i16 0, align 4
@i = common global i32 0, align 4
@l = common global i64 0, align 8
@uc = common global i8 0, align 4
@us = common global i16 0, align 4
@ui = common global i32 0, align 4
@l1 = common global i64 0, align 8

define i64 @func1() nounwind readonly {
entry:
; CHECK-N64: func1
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(c)
; CHECK-N64: lb ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: func1
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(c)
; CHECK-N32: lb ${{[0-9]+}}, 0($[[R0]])
  %0 = load i8, ptr @c, align 4
  %conv = sext i8 %0 to i64
  ret i64 %conv
}

define i64 @func2() nounwind readonly {
entry:
; CHECK-N64: func2
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(s)
; CHECK-N64: lh ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: func2
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(s)
; CHECK-N32: lh ${{[0-9]+}}, 0($[[R0]])
  %0 = load i16, ptr @s, align 4
  %conv = sext i16 %0 to i64
  ret i64 %conv
}

define i64 @func3() nounwind readonly {
entry:
; CHECK-N64: func3
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(i)
; CHECK-N64: lw ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: func3
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(i)
; CHECK-N32: lw ${{[0-9]+}}, 0($[[R0]])
  %0 = load i32, ptr @i, align 4
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

define i64 @func4() nounwind readonly {
entry:
; CHECK-N64: func4
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(l)
; CHECK-N64: ld ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: func4
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(l)
; CHECK-N32: ld ${{[0-9]+}}, 0($[[R0]])
  %0 = load i64, ptr @l, align 8
  ret i64 %0
}

define i64 @ufunc1() nounwind readonly {
entry:
; CHECK-N64: ufunc1
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(uc)
; CHECK-N64: lbu ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: ufunc1
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(uc)
; CHECK-N32: lbu ${{[0-9]+}}, 0($[[R0]])
  %0 = load i8, ptr @uc, align 4
  %conv = zext i8 %0 to i64
  ret i64 %conv
}

define i64 @ufunc2() nounwind readonly {
entry:
; CHECK-N64: ufunc2
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(us)
; CHECK-N64: lhu ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: ufunc2
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(us)
; CHECK-N32: lhu ${{[0-9]+}}, 0($[[R0]])
  %0 = load i16, ptr @us, align 4
  %conv = zext i16 %0 to i64
  ret i64 %conv
}

define i64 @ufunc3() nounwind readonly {
entry:
; CHECK-N64: ufunc3
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(ui)
; CHECK-N64: lwu ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: ufunc3
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(ui)
; CHECK-N32: lwu ${{[0-9]+}}, 0($[[R0]])
  %0 = load i32, ptr @ui, align 4
  %conv = zext i32 %0 to i64
  ret i64 %conv
}

define void @sfunc1() nounwind {
entry:
; CHECK-N64: sfunc1
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(c)
; CHECK-N64: sb ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: sfunc1
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(c)
; CHECK-N32: sb ${{[0-9]+}}, 0($[[R0]])
  %0 = load i64, ptr @l1, align 8
  %conv = trunc i64 %0 to i8
  store i8 %conv, ptr @c, align 4
  ret void
}

define void @sfunc2() nounwind {
entry:
; CHECK-N64: sfunc2
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(s)
; CHECK-N64: sh ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: sfunc2
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(s)
; CHECK-N32: sh ${{[0-9]+}}, 0($[[R0]])
  %0 = load i64, ptr @l1, align 8
  %conv = trunc i64 %0 to i16
  store i16 %conv, ptr @s, align 4
  ret void
}

define void @sfunc3() nounwind {
entry:
; CHECK-N64: sfunc3
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(i)
; CHECK-N64: sw ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: sfunc3
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(i)
; CHECK-N32: sw ${{[0-9]+}}, 0($[[R0]])
  %0 = load i64, ptr @l1, align 8
  %conv = trunc i64 %0 to i32
  store i32 %conv, ptr @i, align 4
  ret void
}

define void @sfunc4() nounwind {
entry:
; CHECK-N64: sfunc4
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(l)
; CHECK-N64: sd ${{[0-9]+}}, 0($[[R0]])
; CHECK-N32: sfunc4
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(l)
; CHECK-N32: sd ${{[0-9]+}}, 0($[[R0]])
  %0 = load i64, ptr @l1, align 8
  store i64 %0, ptr @l, align 8
  ret void
}

