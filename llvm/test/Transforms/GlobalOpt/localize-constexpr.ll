; RUN: opt -S < %s -globalopt | FileCheck %s

@G = internal global i32 42

define i8 @f() norecurse {
; CHECK-LABEL: @f
; CHECK: alloca
; CHECK-NOT: @G
; CHECK: }
  store i32 42, i32* @G
  %a = load i8, i8* bitcast (i32* @G to i8*)
  ret i8 %a
}

@H = internal global i32 42
@Halias = internal alias i32, i32* @H

; @H can't be localized because @Halias uses it, and @Halias can't be converted to an instruction.
define i8 @g() norecurse {
; CHECK-LABEL: @g
; CHECK-NOT: alloca
; CHECK: @H
; CHECK: }
  store i32 42, i32* @H
  %a = load i8, i8* bitcast (i32* @H to i8*)
  ret i8 %a
}

