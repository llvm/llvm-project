; RUN: llc -mtriple=next32 -mcpu=next32gen2 < %s | FileCheck %s
; RUN: llc -mtriple=next32 -filetype=obj -o - %s | llvm-objdump --triple=next32 -d - | FileCheck %s

; Function Attrs: norecurse nounwind readonly willreturn
define dso_local i128 @test_read(i8* nocapture readonly %0, i16* nocapture readonly %1, i32* nocapture readonly %2, i64* nocapture readonly %3, i128* nocapture readonly %4) {
; CHECK-LABEL: test_read
; CHECK:    memread.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r1
; CHECK:    memread.16.align[2] r4, r3, tid
; CHECK-NEXT:    memdata r3
; CHECK:    memread.32.align[4] r6, r5, tid
; CHECK-NEXT:    memdata r2
; CHECK:    memread.64.align[8] r8, r7, tid
; CHECK-NEXT:    memdata r7
; CHECK-NEXT:    memdata r8
; CHECK:    memread.128.align[16] r10, r9, tid
; CHECK-NEXT:    memdata r10
; CHECK-NEXT:    memdata r9
; CHECK-NEXT:    memdata r5
; CHECK-NEXT:    memdata r4
  %6 = load i8, i8* %0, align 1, !tbaa !2
  %7 = zext i8 %6 to i32
  %8 = load i16, i16* %1, align 2, !tbaa !5
  %9 = zext i16 %8 to i32
  %10 = add nuw nsw i32 %9, %7
  %11 = load i32, i32* %2, align 4, !tbaa !7
  %12 = add i32 %10, %11
  %13 = zext i32 %12 to i64
  %14 = load i64, i64* %3, align 8, !tbaa !9
  %15 = add i64 %14, %13
  %16 = zext i64 %15 to i128
  %17 = load i128, i128* %4, align 16, !tbaa !11
  %18 = add i128 %17, %16
  ret i128 %18
}

; Function Attrs: norecurse nounwind readonly willreturn
define dso_local i128 @test_nontemporal_read(i8* nocapture readonly %0, i16* nocapture readonly %1, i32* nocapture readonly %2, i64* nocapture readonly %3, i128* nocapture readonly %4) {
; CHECK-LABEL: test_nontemporal_read
; CHECK:    memread.once.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r1
; CHECK:    memread.once.16.align[2] r4, r3, tid
; CHECK-NEXT:    memdata r3
; CHECK:    memread.once.32.align[4] r6, r5, tid
; CHECK-NEXT:    memdata r2
; CHECK:    memread.once.64.align[8] r8, r7, tid
; CHECK-NEXT:    memdata r7
; CHECK-NEXT:    memdata r8
; CHECK:    memread.once.128.align[16] r10, r9, tid
; CHECK-NEXT:    memdata r10
; CHECK-NEXT:    memdata r9
; CHECK-NEXT:    memdata r5
; CHECK-NEXT:    memdata r4
  %6 = load i8, i8* %0, align 1, !tbaa !2, !nontemporal !13
  %7 = zext i8 %6 to i32
  %8 = load i16, i16* %1, align 2, !tbaa !5, !nontemporal !13
  %9 = zext i16 %8 to i32
  %10 = add nuw nsw i32 %9, %7
  %11 = load i32, i32* %2, align 4, !tbaa !7, !nontemporal !13
  %12 = add i32 %10, %11
  %13 = zext i32 %12 to i64
  %14 = load i64, i64* %3, align 8, !tbaa !9, !nontemporal !13
  %15 = add i64 %14, %13
  %16 = zext i64 %15 to i128
  %17 = load i128, i128* %4, align 16, !tbaa !11, !nontemporal !13
  %18 = add i128 %17, %16
  ret i128 %18
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define dso_local void @test_write(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3, i128* nocapture %4) {
; CHECK-LABEL: test_write
; CHECK:    memwrite.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r11
; CHECK:    memwrite.128.align[16] r10, r9, r1
; CHECK-NEXT:    memdata r12
; CHECK-NEXT:    memdata r12
; CHECK-NEXT:    memdata r12
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r11
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r3
  store i8 8, i8* %0, align 1, !tbaa !2
  store i16 16, i16* %1, align 2, !tbaa !5
  store i32 32, i32* %2, align 4, !tbaa !7
  store i64 64, i64* %3, align 8, !tbaa !9
  store i128 128, i128* %4, align 16, !tbaa !11
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define dso_local void @test_nontemporal_store(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3, i128* nocapture %4) {
; CHECK-LABEL: test_nontemporal_store
; CHECK:    memwrite.once.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r11
; CHECK:    memwrite.once.128.align[16] r10, r9, r1
; CHECK-NEXT:    memdata r12
; CHECK-NEXT:    memdata r12
; CHECK-NEXT:    memdata r12
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.once.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r11
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.once.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.once.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r3
  store i8 8, i8* %0, align 1, !tbaa !2, !nontemporal !13
  store i16 16, i16* %1, align 2, !tbaa !5, !nontemporal !13
  store i32 32, i32* %2, align 4, !tbaa !7, !nontemporal !13
  store i64 64, i64* %3, align 8, !tbaa !9, !nontemporal !13
  store i128 128, i128* %4, align 16, !tbaa !11, !nontemporal !13
  ret void
}

!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !3, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"long", !3, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"__int128", !3, i64 0}
!13 = !{i32 1}
