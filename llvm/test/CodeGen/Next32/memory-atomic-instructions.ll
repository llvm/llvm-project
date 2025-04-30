; RUN: llc -mtriple=next32 -mcpu=next32gen2 < %s | FileCheck %s

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local i128 @test_atomic_read(i8* nocapture readonly %0, i16* nocapture readonly %1, i32* nocapture readonly %2, i64* nocapture readonly %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_read
; CHECK:    memread.atomic.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r1
; CHECK:    memread.atomic.16.align[2] r4, r3, r2
; CHECK-NEXT:    memdata r3
; CHECK:    memread.atomic.32.align[4] r6, r5, r2
; CHECK-NEXT:    memdata r1
; CHECK:    memread.atomic.64.align[8] r8, r7, r2
; CHECK-NEXT:    memdata r1
; CHECK-NEXT:    memdata r4
  %5 = load atomic i8, i8* %0 seq_cst, align 1, !tbaa !2
  %6 = zext i8 %5 to i32
  %7 = load atomic i16, i16* %1 seq_cst, align 2, !tbaa !2
  %8 = zext i16 %7 to i32
  %9 = add nuw nsw i32 %8, %6
  %10 = load atomic i32, i32* %2 seq_cst, align 4, !tbaa !2
  %11 = add i32 %9, %10
  %12 = zext i32 %11 to i64
  %13 = load atomic i64, i64* %3 seq_cst, align 8, !tbaa !2
  %14 = add i64 %13, %12
  %15 = zext i64 %14 to i128
  ret i128 %15
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_write(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_write
; CHECK:    memwrite.atomic.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memwrite.atomic.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.atomic.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memwrite.atomic.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  store atomic i8 8, i8* %0 seq_cst, align 1, !tbaa !2
  store atomic i16 16, i16* %1 seq_cst, align 2, !tbaa !2
  store atomic i32 32, i32* %2 seq_cst, align 4, !tbaa !2
  store atomic i64 64, i64* %3 seq_cst, align 8, !tbaa !2
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_fetch_add(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_fetch_add
; CHECK:    memfa.add.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memfa.add.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.add.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.add.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  %5 = atomicrmw add i8* %0, i8 8 seq_cst
  %6 = atomicrmw add i16* %1, i16 16 seq_cst
  %7 = atomicrmw add i32* %2, i32 32 seq_cst
  %8 = atomicrmw add i64* %3, i64 64 seq_cst
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_fetch_sub(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_fetch_sub
; CHECK:    memfa.sub.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memfa.sub.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.sub.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.sub.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  %5 = atomicrmw sub i8* %0, i8 8 seq_cst
  %6 = atomicrmw sub i16* %1, i16 16 seq_cst
  %7 = atomicrmw sub i32* %2, i32 32 seq_cst
  %8 = atomicrmw sub i64* %3, i64 64 seq_cst
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_fetch_or(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_fetch_or
; CHECK:    memfa.or.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memfa.or.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.or.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.or.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  %5 = atomicrmw or i8* %0, i8 8 seq_cst
  %6 = atomicrmw or i16* %1, i16 16 seq_cst
  %7 = atomicrmw or i32* %2, i32 32 seq_cst
  %8 = atomicrmw or i64* %3, i64 64 seq_cst
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_fetch_xor(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_fetch_xor
; CHECK:    memfa.xor.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memfa.xor.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.xor.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.xor.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  %5 = atomicrmw xor i8* %0, i8 8 seq_cst
  %6 = atomicrmw xor i16* %1, i16 16 seq_cst
  %7 = atomicrmw xor i32* %2, i32 32 seq_cst
  %8 = atomicrmw xor i64* %3, i64 64 seq_cst
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_fetch_and(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_fetch_and
; CHECK:    memfa.and.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memfa.and.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.and.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.and.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  %5 = atomicrmw and i8* %0, i8 8 seq_cst
  %6 = atomicrmw and i16* %1, i16 16 seq_cst
  %7 = atomicrmw and i32* %2, i32 32 seq_cst
  %8 = atomicrmw and i64* %3, i64 64 seq_cst
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_exchange(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_exchange
; CHECK:    memfa.set.8.align[1] r2, r1, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memfa.set.16.align[2] r4, r3, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.set.32.align[4] r6, r5, r1
; CHECK-NEXT:    memdata r2
; CHECK:    memfa.set.64.align[8] r8, r7, r1
; CHECK-NEXT:    memdata r3
; CHECK-NEXT:    memdata r2
  %5 = atomicrmw xchg i8* %0, i8 8 seq_cst
  %6 = atomicrmw xchg i16* %1, i16 16 seq_cst
  %7 = atomicrmw xchg i32* %2, i32 32 seq_cst
  %8 = atomicrmw xchg i64* %3, i64 64 seq_cst
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local void @test_atomic_compare_exchange(i8* nocapture %0, i16* nocapture %1, i32* nocapture %2, i64* nocapture %3, i128* nocapture readnone %4, i64* nocapture %5, i64 %6) local_unnamed_addr #0 {
; CHECK-LABEL: test_atomic_compare_exchange
; CHECK:    memread.8.align[1] r12, r11, tid
; CHECK-NEXT:    memdata r10
; CHECK:    memcas.8.align[1] r2, r1, r15
; CHECK-NEXT:    memdata r10
; CHECK-NEXT:    memdata r9
; CHECK:    memwrite.8.align[1] r12, r11, tid
; CHECK-NEXT:    memdata r9
; CHECK:    memread.16.align[2] r12, r11, tid
; CHECK-NEXT:    memdata r2
; CHECK:    memcas.16.align[2] r4, r3, r9
; CHECK-NEXT:    memdata r2
; CHECK-NEXT:    memdata r1
; CHECK:    memwrite.16.align[2] r12, r11, tid
; CHECK-NEXT:    memdata r1
; CHECK:    memread.32.align[4] r12, r11, tid
; CHECK-NEXT:    memdata r2
; CHECK:    memcas.32.align[4] r6, r5, r3
; CHECK-NEXT:    memdata r2
; CHECK-NEXT:    memdata r1
; CHECK:    memwrite.32.align[4] r12, r11, tid
; CHECK-NEXT:    memdata r1
; CHECK:    memread.64.align[8] r12, r11, tid
; CHECK-NEXT:    memdata r1
; CHECK-NEXT:    memdata r2
; CHECK:    memcas.64.align[8] r8, r7, r3
; CHECK-NEXT:    memdata r1
; CHECK-NEXT:    memdata r2
; CHECK-NEXT:    memdata r14
; CHECK-NEXT:    memdata r13
; CHECK:    memwrite.64.align[8] r12, r11, tid
; CHECK-NEXT:    memdata r14
; CHECK-NEXT:    memdata r13
  %8 = bitcast i64* %5 to i8*
  %9 = trunc i64 %6 to i8
  %10 = load i8, i8* %8, align 1
  %11 = cmpxchg i8* %0, i8 %10, i8 %9 seq_cst seq_cst
  %12 = extractvalue { i8, i1 } %11, 1
  br i1 %12, label %15, label %13

13:                                               ; preds = %7
  %14 = extractvalue { i8, i1 } %11, 0
  store i8 %14, i8* %8, align 1
  br label %15

15:                                               ; preds = %13, %7
  %16 = bitcast i64* %5 to i16*
  %17 = trunc i64 %6 to i16
  %18 = load i16, i16* %16, align 2
  %19 = cmpxchg i16* %1, i16 %18, i16 %17 seq_cst seq_cst
  %20 = extractvalue { i16, i1 } %19, 1
  br i1 %20, label %23, label %21

21:                                               ; preds = %15
  %22 = extractvalue { i16, i1 } %19, 0
  store i16 %22, i16* %16, align 2
  br label %23

23:                                               ; preds = %21, %15
  %24 = bitcast i64* %5 to i32*
  %25 = trunc i64 %6 to i32
  %26 = load i32, i32* %24, align 4
  %27 = cmpxchg i32* %2, i32 %26, i32 %25 seq_cst seq_cst
  %28 = extractvalue { i32, i1 } %27, 1
  br i1 %28, label %31, label %29

29:                                               ; preds = %23
  %30 = extractvalue { i32, i1 } %27, 0
  store i32 %30, i32* %24, align 4
  br label %31

31:                                               ; preds = %29, %23
  %32 = load i64, i64* %5, align 8
  %33 = cmpxchg i64* %3, i64 %32, i64 %6 seq_cst seq_cst
  %34 = extractvalue { i64, i1 } %33, 1
  br i1 %34, label %37, label %35

35:                                               ; preds = %31
  %36 = extractvalue { i64, i1 } %33, 0
  store i64 %36, i64* %5, align 8
  br label %37

37:                                               ; preds = %35, %31
  ret void
}

!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
