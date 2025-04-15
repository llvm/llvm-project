; RUN: llc -mtriple=bpfel -mcpu=v1 < %s | FileCheck --check-prefix=CHECK-V1 %s
; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck --check-prefix=CHECK-V4 %s

target triple = "bpf"

;   struct S {
;     int var[3];
;   };
;   int foo1 (struct S *a, struct S *b)
;   {
;     return a - b;
;   }
define dso_local i32 @foo1(ptr noundef %a, ptr noundef %b) local_unnamed_addr {
entry:
  %sub.ptr.lhs.cast = ptrtoint ptr %a to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %b to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = sdiv exact i64 %sub.ptr.sub, 12
  %conv = trunc i64 %sub.ptr.div to i32
  ret i32 %conv
}
; CHECK-V1:        r0 = r1
; CHECK-V1:        r0 -= r2
; CHECK-V1:        r0 s>>= 2
; CHECK-V1:        r1 = -6148914691236517205 ll
; CHECK-V1:        r0 *= r1
; CHECK-V1:        exit

; CHECK-V4:        r0 = r1
; CHECK-V4:        r0 -= r2
; CHECK-V4:        r0 >>= 2
; CHECK-V4:        w0 *= -1431655765
; CHECK-V4:        exit

define dso_local noundef range(i32 -143165576, 143165577) i32 @foo2(i32 noundef %a) local_unnamed_addr {
entry:
  %div = sdiv i32 %a, 15
  ret i32 %div
}
; CHECK-V1-NOT:   r[[#]] s/= 15
; CHECK-V4-NOT:   w[[#]] s/= 15

define dso_local noundef range(i32 -14, 15) i32 @foo3(i32 noundef %a) local_unnamed_addr {
entry:
  %rem = srem i32 %a, 15
  ret i32 %rem
}
; CHECK-V1-NOT:   r[[#]] s%= 15
; CHECK-V4-NOT:   w[[#]] s%= 15

define dso_local i64 @foo4(i64 noundef %a) local_unnamed_addr {
entry:
  %div = udiv exact i64 %a, 15
  ret i64 %div
}
; CHECK-V1-NOT:   r[[#]] /= 15
; CHECK-V4-NOT:   w[[#]] /= 15
