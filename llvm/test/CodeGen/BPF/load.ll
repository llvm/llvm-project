; RUN: llc < %s -march=bpfel -mcpu=v1 | FileCheck %s

define i16 @am1(ptr %a) nounwind {
  %1 = load i16, ptr %a
  ret i16 %1
}
; CHECK-LABEL: am1:
; CHECK: r0 = *(u16 *)(r1 + 0)

@foo = external global i16

define i16 @am2() nounwind {
  %1 = load i16, ptr @foo
  ret i16 %1
}
; CHECK-LABEL: am2:
; CHECK: r0 = *(u16 *)(r1 + 0)

define i16 @am4() nounwind {
  %1 = load volatile i16, ptr inttoptr(i16 32 to ptr)
  ret i16 %1
}
; CHECK-LABEL: am4:
; CHECK: r1 = 32
; CHECK: r0 = *(u16 *)(r1 + 0)

define i16 @am5(ptr %a) nounwind {
  %1 = getelementptr i16, ptr %a, i16 2
  %2 = load i16, ptr %1
  ret i16 %2
}
; CHECK-LABEL: am5:
; CHECK: r0 = *(u16 *)(r1 + 4)

%S = type { i16, i16 }
@baz = common global %S zeroinitializer, align 1

define i16 @am6() nounwind {
  %1 = load i16, ptr getelementptr (%S, ptr @baz, i32 0, i32 1)
  ret i16 %1
}
; CHECK-LABEL: am6:
; CHECK: r0 = *(u16 *)(r1 + 2)
