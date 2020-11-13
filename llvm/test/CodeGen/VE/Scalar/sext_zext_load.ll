; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i16 @func1() {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = sext i8 %a.val to i16
  ret i16 %a.conv
}

define i32 @func2() {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = sext i8 %a.val to i32
  ret i32 %a.conv
}

define i64 @func3() {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = sext i8 %a.val to i64
  ret i64 %a.conv
}

define zeroext i16 @func5() {
; CHECK-LABEL: func5:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 191(, %s11)
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = sext i8 %a.val to i16
  ret i16 %a.conv
}

define i32 @func6() {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = sext i8 %a.val to i32
  ret i32 %a.conv
}

define i64 @func7() {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = sext i8 %a.val to i64
  ret i64 %a.conv
}

define signext i16 @func9() {
; CHECK-LABEL: func9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = zext i8 %a.val to i16
  ret i16 %a.conv
}

define i32 @func10() {
; CHECK-LABEL: func10:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = zext i8 %a.val to i32
  ret i32 %a.conv
}

define i64 @func11() {
; CHECK-LABEL: func11:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = zext i8 %a.val to i64
  ret i64 %a.conv
}

define zeroext i16 @func13() {
; CHECK-LABEL: func13:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = zext i8 %a.val to i16
  ret i16 %a.conv
}

define zeroext i16 @func14() {
; CHECK-LABEL: func14:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = zext i8 %a.val to i16
  ret i16 %a.conv
}

define i64 @func15() {
; CHECK-LABEL: func15:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i8, align 1
  %a.val = load i8, i8* %a, align 1
  %a.conv = zext i8 %a.val to i64
  ret i64 %a.conv
}

define i32 @func17() {
; CHECK-LABEL: func17:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.sx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.val = load i16, i16* %a, align 2
  %a.conv = sext i16 %a.val to i32
  ret i32 %a.conv
}

define i64 @func18() {
; CHECK-LABEL: func18:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.sx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.val = load i16, i16* %a, align 2
  %a.conv = sext i16 %a.val to i64
  ret i64 %a.conv
}

define zeroext i16 @func20() {
; CHECK-LABEL: func20:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.conv = load i16, i16* %a, align 2
  ret i16 %a.conv
}

define i64 @func21() {
; CHECK-LABEL: func21:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.sx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.val = load i16, i16* %a, align 2
  %a.conv = sext i16 %a.val to i64
  ret i64 %a.conv
}

define i32 @func23() {
; CHECK-LABEL: func23:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.val = load i16, i16* %a, align 2
  %a.conv = zext i16 %a.val to i32
  ret i32 %a.conv
}

define i64 @func24() {
; CHECK-LABEL: func24:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.val = load i16, i16* %a, align 2
  %a.conv = zext i16 %a.val to i64
  ret i64 %a.conv
}

define zeroext i16 @func26() {
; CHECK-LABEL: func26:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.conv = load i16, i16* %a, align 2
  ret i16 %a.conv
}

define i64 @func27() {
; CHECK-LABEL: func27:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 190(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i16, align 2
  %a.val = load i16, i16* %a, align 2
  %a.conv = zext i16 %a.val to i64
  ret i64 %a.conv
}

define i64 @func29() {
; CHECK-LABEL: func29:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, 188(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i32, align 4
  %a.val = load i32, i32* %a, align 4
  %a.conv = sext i32 %a.val to i64
  ret i64 %a.conv
}

define i64 @func31() {
; CHECK-LABEL: func31:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, 188(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i32, align 4
  %a.val = load i32, i32* %a, align 4
  %a.conv = sext i32 %a.val to i64
  ret i64 %a.conv
}

define i64 @func33() {
; CHECK-LABEL: func33:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.zx %s0, 188(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i32, align 4
  %a.val = load i32, i32* %a, align 4
  %a.conv = zext i32 %a.val to i64
  ret i64 %a.conv
}

define i64 @func35() {
; CHECK-LABEL: func35:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.zx %s0, 188(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i32, align 4
  %a.val = load i32, i32* %a, align 4
  %a.conv = zext i32 %a.val to i64
  ret i64 %a.conv
}

define signext i8 @func37() {
; CHECK-LABEL: func37:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    sll %s0, %s0, 63
; CHECK-NEXT:    sra.l %s0, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = sext i1 %a.val to i8
  ret i8 %a.conv
}

define signext i16 @func38() {
; CHECK-LABEL: func38:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    sll %s0, %s0, 63
; CHECK-NEXT:    sra.l %s0, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = sext i1 %a.val to i16
  ret i16 %a.conv
}

define signext i32 @func39() {
; CHECK-LABEL: func39:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    sll %s0, %s0, 63
; CHECK-NEXT:    sra.l %s0, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = sext i1 %a.val to i32
  ret i32 %a.conv
}

define signext i64 @func40() {
; CHECK-LABEL: func40:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    sll %s0, %s0, 63
; CHECK-NEXT:    sra.l %s0, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = sext i1 %a.val to i64
  ret i64 %a.conv
}

define signext i8 @func42() {
; CHECK-LABEL: func42:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = zext i1 %a.val to i8
  ret i8 %a.conv
}

define signext i16 @func43() {
; CHECK-LABEL: func43:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = zext i1 %a.val to i16
  ret i16 %a.conv
}

define signext i32 @func44() {
; CHECK-LABEL: func44:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = zext i1 %a.val to i32
  ret i32 %a.conv
}

define signext i64 @func45() {
; CHECK-LABEL: func45:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i1, align 1
  %a.val = load i1, i1* %a, align 1
  %a.conv = zext i1 %a.val to i64
  ret i64 %a.conv
}
