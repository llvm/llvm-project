; RUN: llc -mtriple=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC16

@s = global i8 115, align 1
@c = common global i8 0, align 1
@.str = private unnamed_addr constant [5 x i8] c"%c \0A\00", align 1

define void @test(i32 %i) nounwind {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 115, label %sw.bb
    i32 105, label %sw.bb1
    i32 100, label %sw.bb2
    i32 108, label %sw.bb3
    i32 99, label %sw.bb4
    i32 68, label %sw.bb5
    i32 81, label %sw.bb6
    i32 76, label %sw.bb7
  ]

sw.bb:                                            ; preds = %entry
  store i8 115, i8* @c, align 1
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  store i8 105, i8* @c, align 1
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  store i8 100, i8* @c, align 1
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  store i8 108, i8* @c, align 1
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  store i8 99, i8* @c, align 1
  br label %sw.epilog

sw.bb5:                                           ; preds = %entry
  store i8 68, i8* @c, align 1
  br label %sw.epilog

sw.bb6:                                           ; preds = %entry
  store i8 81, i8* @c, align 1
  br label %sw.epilog

sw.bb7:                                           ; preds = %entry
  store i8 76, i8* @c, align 1
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb7, %sw.bb6, %sw.bb5, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  ret void
}

; CHECK-STATIC16: li	${{[0-9]+}}, %hi($JTI{{[0-9]+}}_{{[0-9]+}})
; CHECK-STATIC16: lw	${{[0-9]+}}, %lo($JTI{{[0-9]+}}_{{[0-9]+}})(${{[0-9]+}})
; CHECK-STATIC16: $JTI{{[0-9]+}}_{{[0-9]+}}:
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
; CHECK-STATIC16: .4byte ($BB0_{{[0-9]+}})
