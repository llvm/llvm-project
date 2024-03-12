; RUN: llc -march=bpfel -mcpu=v1 < %s | FileCheck --check-prefix=CHECK-V1 %s
; RUN: llc -march=bpfel -mcpu=v2 < %s | FileCheck --check-prefix=CHECK-V2 %s
; RUN: llc -march=bpfel -mcpu=v3 < %s | FileCheck --check-prefix=CHECK-V3 %s
; RUN: llc -march=bpfel -mcpu=v4 < %s | FileCheck --check-prefix=CHECK-V4 %s

declare dso_local void @bar(i16 noundef signext, i32 noundef signext, i32 noundef signext) local_unnamed_addr

define void @test() {
entry:
  tail call void @bar(i16 noundef signext -3, i32 noundef signext 4, i32 noundef signext -5)
; CHECK-V1:     r2 = 4
; CHECK-V1:     r3 = -5
; CHECK-V2:     r2 = 4
; CHECK-V2:     r3 = -5
; CHECK-V3:     w2 = 4
; CHECK-V3:     w3 = -5
; CHECK-V4:     w2 = 4
; CHECK-V4:     w3 = -5
  ret void
}

define dso_local void @test2(i64 noundef %a, i64 noundef %b) local_unnamed_addr {
entry:
  %conv = trunc i64 %a to i16
  %conv1 = trunc i64 %b to i32
  tail call void @bar(i16 noundef signext %conv, i32 noundef signext %conv1, i32 noundef signext %conv1)
; CHECK-V1:     r2 <<= 32
; CHECK-V1:     r2 s>>= 32
; CHECK-V2:     r2 <<= 32
; CHECK-V2:     r2 s>>= 32
; CHECK-V3-NOT: r2 <<= 32
; CHECK-V4-NOT: r2 <<= 32
  ret void
}
