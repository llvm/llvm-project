; RUN: llc  -mtriple=aarch64-linux-gnu %s | FileCheck %s
;
; Test ensures that the compiler generates no extra instructions
; for __builtin_clzg output type conversion
;
; IR for this test was generated from the following source code:
; #include <stdint.h>
; int32_t foo8(uint8_t x) { return __builtin_clzg(x); }
; int32_t foo16(uint16_t x) { return __builtin_clzg(x); }

; CHECK-LABEL: foo8:
; CHECK: %bb.0:
; CHECK-NEXT: and w8, w0, #0xff
; CHECK-NEXT: clz w8, w8
; CHECK-NEXT: sub w0, w8, #24
; CHECK-NEXT: ret
define dso_local range(i32 0, 9) i32 @foo8(i8 noundef %0) local_unnamed_addr #0 {
  %2 = tail call range(i8 0, 9) i8 @llvm.ctlz.i8(i8 %0, i1 false)
  %3 = zext nneg i8 %2 to i32
  ret i32 %3
}

; CHECK-LABEL: foo16:
; CHECK: %bb.0:
; CHECK-NEXT: and w8, w0, #0xffff
; CHECK-NEXT: clz w8, w8
; CHECK-NEXT: sub w0, w8, #16
; CHECK-NEXT: ret
define dso_local range(i32 0, 17) i32 @foo16(i16 noundef %0) local_unnamed_addr #0 {
  %2 = tail call range(i16 0, 17) i16 @llvm.ctlz.i16(i16 %0, i1 false)
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 4}
!5 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git)"}
