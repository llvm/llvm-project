; The reason for the bug was that when deciding if a global
; variable can be part of sdata, we were wrongly ignoring
; the presence of any section specified for the variable
; using the section attribute. If such a section is specified,
; and that section is not sdata*/sbss* then the variable
; cannot use GPREL addressing, i.e. memw(#variablename).

; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-LABEL: foo
; CHECK-DAG: memw(##b)
; CHECK-DAG: memw(gp+#d)
; CHECK-DAG: memw(##g)
; CHECK-DAG: memw(gp+#h)
; CHECK-DAG: memw(gp+#f)
; CHECK-DAG: memw(##e)
; CHECK-DAG: memw(gp+#a)
; CHECK-DAG: memw(gp+#c)
; CHECK-LABEL: bar
; CHECK: memw(##b)

@b = global i32 0, section ".data.section", align 4
@a = common global i32 0, align 4
@d = global i32 0, section ".sbss", align 4
@c = global i32 0, section ".sdata", align 4
@f = global i32 0, section ".sbss.4", align 4
@e = global i32 0, section ".sdatafoo", align 4
@h = global i32 0, section ".sdata.4", align 4
@g = global i32 0, section ".sbssfoo", align 4

define void @foo() nounwind {
entry:
  %0 = load i32, i32* @b, align 4
  store i32 %0, i32* @a, align 4
  %1 = load i32, i32* @d, align 4
  store i32 %1, i32* @c, align 4
  %2 = load i32, i32* @f, align 4
  store i32 %2, i32* @e, align 4
  %3 = load i32, i32* @h, align 4
  store i32 %3, i32* @g, align 4
  ret void
}

define void @bar() nounwind section ".function.section" {
entry:
  %0 = load i32, i32* @a, align 4
  store i32 %0, i32* @b, align 4
  ret void
}

define i32 @main() nounwind readnone {
entry:
  ret i32 0
}

