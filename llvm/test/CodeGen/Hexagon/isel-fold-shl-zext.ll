; RUN: llc -mtriple=hexagon-unknown-elf < %s | FileCheck %s

; In ISelLowering, when folding nodes (or (shl xx, s), (zext y))
; to (COMBINE (shl xx, s-32), y) where s >= 32,
; check that resulting shift value does not create an undef


target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nofree nosync nounwind memory(readwrite, inaccessiblemem: none)
define dso_local void @foo(i64* nocapture noundef %buf, i32 %a, i32 %b) local_unnamed_addr {
; CHECK-LABEL: foo:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    {
; CHECK-NEXT:     r[[REG0:[0-9]+]] = addasl(r2,r1,#1)
; CHECK-NEXT:     r[[REG2:[0-9]+]] = asl(r1,#1)
; CHECK-NEXT:    }
; CHECK-NEXT:    {
; CHECK-NEXT:     r[[REG1:[0-9]+]] = addasl(r[[REG0]],r1,#1)
; CHECK-NEXT:    }
; CHECK-NEXT:    {
; CHECK-NEXT:     jumpr r31
; CHECK-NEXT:     memd(r0+#8) = r[[REG2]]:[[REG1]]
; CHECK-NEXT:    }
entry:
  %arrayidx = getelementptr inbounds i64, i64* %buf, i32 1
  %add0 = shl nsw i32 %a, 1
  %add1 = add nsw i32 %add0, %b
  %add2 = add nsw i32 %add1, %add0
  %ext0 = zext i32 %add0 to i64
  %shift0 = shl nuw i64 %ext0, 32
  %ext1 = zext i32 %add2 to i64
  %or0 = or i64 %shift0, %ext1
  store i64 %or0, i64* %arrayidx, align 8
  ret void
}
