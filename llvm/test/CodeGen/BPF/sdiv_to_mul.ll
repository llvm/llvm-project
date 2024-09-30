; RUN: llc -O2 -march=bpfel -mcpu=v1 < %s | FileCheck --check-prefix=CHECK-V1 %s
; RUN: llc -O2 -march=bpfel -mcpu=v3 < %s | FileCheck --check-prefix=CHECK-V3 %s
; RUN: llc -O2 -march=bpfel -mcpu=v4 < %s | FileCheck --check-prefix=CHECK-V4 %s
;
; Source:
;   struct S {
;     int var[3];
;   };
;   int foo1 (struct S *a, struct S *b)
;   {
;     return a - b;
;   }
;   int foo2(int a)
;   {
;     return a/15;
;   }
;   int foo3(int a)
;   {
;     return a%15;
;   }

target triple = "bpf"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i32 @foo1(ptr noundef %a, ptr noundef %b) local_unnamed_addr #0 {
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

; CHECK-V3:        r0 = r1
; CHECK-V3:        r0 -= r2
; CHECK-V3:        r0 >>= 2
; CHECK-V3:        w0 *= -1431655765
; CHECK-V3:        exit

; CHECK-V4:        r0 = r1
; CHECK-V4:        r0 -= r2
; CHECK-V4:        r0 s/= 12
; CHECK-V4:        exit

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local noundef range(i32 -143165576, 143165577) i32 @foo2(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %div = sdiv i32 %a, 15
  ret i32 %div
}
; CHECK-V1-NOT:   r[[#]] s/= 15
; CHECK-V3-NOT:   w[[#]] s/= 15
; CHECK-V4:       w[[#]] s/= 15

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local noundef range(i32 -14, 15) i32 @foo3(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %rem = srem i32 %a, 15
  ret i32 %rem
}

; CHECK-V1-NOT:   r[[#]] s%= 15
; CHECK-V3-NOT:   w[[#]] s%= 15
; CHECK-V4:       w[[#]] s%= 15

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v1" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 20.0.0git (git@github.com:yonghong-song/llvm-project.git 238f3f994a96c511134ca1bc11d2d03e4368a0c1)"}
