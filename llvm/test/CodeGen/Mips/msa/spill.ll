; Test that the correct instruction is chosen for spill and reload by trying
; to have 33 live MSA registers simultaneously

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

define i32 @test_i8(<16 x i8>* %p0, <16 x i8>* %q1) nounwind {
entry:
  %p1  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 1
  %p2  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 2
  %p3  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 3
  %p4  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 4
  %p5  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 5
  %p6  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 6
  %p7  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 7
  %p8  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 8
  %p9  = getelementptr <16 x i8>, <16 x i8>* %p0, i32 9
  %p10 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 10
  %p11 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 11
  %p12 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 12
  %p13 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 13
  %p14 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 14
  %p15 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 15
  %p16 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 16
  %p17 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 17
  %p18 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 18
  %p19 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 19
  %p20 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 20
  %p21 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 21
  %p22 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 22
  %p23 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 23
  %p24 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 24
  %p25 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 25
  %p26 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 26
  %p27 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 27
  %p28 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 28
  %p29 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 29
  %p30 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 30
  %p31 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 31
  %p32 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 32
  %p33 = getelementptr <16 x i8>, <16 x i8>* %p0, i32 33
  %0  = load <16 x i8>, <16 x i8>* %p0, align 16
  %1  = load <16 x i8>, <16 x i8>* %p1, align 16
  %2  = load <16 x i8>, <16 x i8>* %p2, align 16
  %3  = load <16 x i8>, <16 x i8>* %p3, align 16
  %4  = load <16 x i8>, <16 x i8>* %p4, align 16
  %5  = load <16 x i8>, <16 x i8>* %p5, align 16
  %6  = load <16 x i8>, <16 x i8>* %p6, align 16
  %7  = load <16 x i8>, <16 x i8>* %p7, align 16
  %8  = load <16 x i8>, <16 x i8>* %p8, align 16
  %9  = load <16 x i8>, <16 x i8>* %p9, align 16
  %10 = load <16 x i8>, <16 x i8>* %p10, align 16
  %11 = load <16 x i8>, <16 x i8>* %p11, align 16
  %12 = load <16 x i8>, <16 x i8>* %p12, align 16
  %13 = load <16 x i8>, <16 x i8>* %p13, align 16
  %14 = load <16 x i8>, <16 x i8>* %p14, align 16
  %15 = load <16 x i8>, <16 x i8>* %p15, align 16
  %16 = load <16 x i8>, <16 x i8>* %p16, align 16
  %17 = load <16 x i8>, <16 x i8>* %p17, align 16
  %18 = load <16 x i8>, <16 x i8>* %p18, align 16
  %19 = load <16 x i8>, <16 x i8>* %p19, align 16
  %20 = load <16 x i8>, <16 x i8>* %p20, align 16
  %21 = load <16 x i8>, <16 x i8>* %p21, align 16
  %22 = load <16 x i8>, <16 x i8>* %p22, align 16
  %23 = load <16 x i8>, <16 x i8>* %p23, align 16
  %24 = load <16 x i8>, <16 x i8>* %p24, align 16
  %25 = load <16 x i8>, <16 x i8>* %p25, align 16
  %26 = load <16 x i8>, <16 x i8>* %p26, align 16
  %27 = load <16 x i8>, <16 x i8>* %p27, align 16
  %28 = load <16 x i8>, <16 x i8>* %p28, align 16
  %29 = load <16 x i8>, <16 x i8>* %p29, align 16
  %30 = load <16 x i8>, <16 x i8>* %p30, align 16
  %31 = load <16 x i8>, <16 x i8>* %p31, align 16
  %32 = load <16 x i8>, <16 x i8>* %p32, align 16
  %33 = load <16 x i8>, <16 x i8>* %p33, align 16
  %r1  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0,   <16 x i8> %1)
  %r2  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r1,  <16 x i8> %2)
  %r3  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r2,  <16 x i8> %3)
  %r4  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r3,  <16 x i8> %4)
  %r5  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r4,  <16 x i8> %5)
  %r6  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r5,  <16 x i8> %6)
  %r7  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r6,  <16 x i8> %7)
  %r8  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r7,  <16 x i8> %8)
  %r9  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r8,  <16 x i8> %9)
  %r10 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r9,  <16 x i8> %10)
  %r11 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r10, <16 x i8> %11)
  %r12 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r11, <16 x i8> %12)
  %r13 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r12, <16 x i8> %13)
  %r14 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r13, <16 x i8> %14)
  %r15 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r14, <16 x i8> %15)
  %r16 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r15, <16 x i8> %16)
  %r17 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r16, <16 x i8> %17)
  %r18 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r17, <16 x i8> %18)
  %r19 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r18, <16 x i8> %19)
  %r20 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r19, <16 x i8> %20)
  %r21 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r20, <16 x i8> %21)
  %r22 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r21, <16 x i8> %22)
  %r23 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r22, <16 x i8> %23)
  %r24 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r23, <16 x i8> %24)
  %r25 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r24, <16 x i8> %25)
  %r26 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r25, <16 x i8> %26)
  %r27 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r26, <16 x i8> %27)
  %r28 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r27, <16 x i8> %28)
  %r29 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r28, <16 x i8> %29)
  %r30 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r29, <16 x i8> %30)
  %r31 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r30, <16 x i8> %31)
  %r32 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r31, <16 x i8> %32)
  %r33 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r32, <16 x i8> %33)
  %rx1  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %r33,   <16 x i8> %1)
  %rx2  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx1,  <16 x i8> %2)
  %rx3  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx2,  <16 x i8> %3)
  %rx4  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx3,  <16 x i8> %4)
  %rx5  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx4,  <16 x i8> %5)
  %rx6  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx5,  <16 x i8> %6)
  %rx7  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx6,  <16 x i8> %7)
  %rx8  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx7,  <16 x i8> %8)
  %rx9  = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx8,  <16 x i8> %9)
  %rx10 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx9,  <16 x i8> %10)
  %rx11 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx10, <16 x i8> %11)
  %rx12 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx11, <16 x i8> %12)
  %rx13 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx12, <16 x i8> %13)
  %rx14 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx13, <16 x i8> %14)
  %rx15 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx14, <16 x i8> %15)
  %rx16 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx15, <16 x i8> %16)
  %rx17 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx16, <16 x i8> %17)
  %rx18 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx17, <16 x i8> %18)
  %rx19 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx18, <16 x i8> %19)
  %rx20 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx19, <16 x i8> %20)
  %rx21 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx20, <16 x i8> %21)
  %rx22 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx21, <16 x i8> %22)
  %rx23 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx22, <16 x i8> %23)
  %rx24 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx23, <16 x i8> %24)
  %rx25 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx24, <16 x i8> %25)
  %rx26 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx25, <16 x i8> %26)
  %rx27 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx26, <16 x i8> %27)
  %rx28 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx27, <16 x i8> %28)
  %rx29 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx28, <16 x i8> %29)
  %rx30 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx29, <16 x i8> %30)
  %rx31 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx30, <16 x i8> %31)
  %rx32 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx31, <16 x i8> %32)
  %rx33 = call <16 x i8> @llvm.mips.addv.b(<16 x i8> %rx32, <16 x i8> %33)
  %res = call i32 @llvm.mips.copy.s.b(<16 x i8> %rx33, i32 0)
  ret i32 %res
}

declare <16 x i8> @llvm.mips.addv.b(<16 x i8>, <16 x i8>) nounwind
declare i32       @llvm.mips.copy.s.b(<16 x i8>, i32) nounwind

; CHECK: test_i8:
; CHECK: st.b {{.*}} Spill
; CHECK: st.b {{.*}} Spill
; CHECK: ld.b {{.*}} Reload
; CHECK: ld.b {{.*}} Reload
; CHECK: .size

define i32 @test_i16(<8 x i16>* %p0, <8 x i16>* %q1) nounwind {
entry:
  %p1  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 1
  %p2  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 2
  %p3  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 3
  %p4  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 4
  %p5  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 5
  %p6  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 6
  %p7  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 7
  %p8  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 8
  %p9  = getelementptr <8 x i16>, <8 x i16>* %p0, i32 9
  %p10 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 10
  %p11 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 11
  %p12 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 12
  %p13 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 13
  %p14 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 14
  %p15 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 15
  %p16 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 16
  %p17 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 17
  %p18 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 18
  %p19 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 19
  %p20 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 20
  %p21 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 21
  %p22 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 22
  %p23 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 23
  %p24 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 24
  %p25 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 25
  %p26 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 26
  %p27 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 27
  %p28 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 28
  %p29 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 29
  %p30 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 30
  %p31 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 31
  %p32 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 32
  %p33 = getelementptr <8 x i16>, <8 x i16>* %p0, i32 33
  %0  = load <8 x i16>, <8 x i16>* %p0, align 16
  %1  = load <8 x i16>, <8 x i16>* %p1, align 16
  %2  = load <8 x i16>, <8 x i16>* %p2, align 16
  %3  = load <8 x i16>, <8 x i16>* %p3, align 16
  %4  = load <8 x i16>, <8 x i16>* %p4, align 16
  %5  = load <8 x i16>, <8 x i16>* %p5, align 16
  %6  = load <8 x i16>, <8 x i16>* %p6, align 16
  %7  = load <8 x i16>, <8 x i16>* %p7, align 16
  %8  = load <8 x i16>, <8 x i16>* %p8, align 16
  %9  = load <8 x i16>, <8 x i16>* %p9, align 16
  %10 = load <8 x i16>, <8 x i16>* %p10, align 16
  %11 = load <8 x i16>, <8 x i16>* %p11, align 16
  %12 = load <8 x i16>, <8 x i16>* %p12, align 16
  %13 = load <8 x i16>, <8 x i16>* %p13, align 16
  %14 = load <8 x i16>, <8 x i16>* %p14, align 16
  %15 = load <8 x i16>, <8 x i16>* %p15, align 16
  %16 = load <8 x i16>, <8 x i16>* %p16, align 16
  %17 = load <8 x i16>, <8 x i16>* %p17, align 16
  %18 = load <8 x i16>, <8 x i16>* %p18, align 16
  %19 = load <8 x i16>, <8 x i16>* %p19, align 16
  %20 = load <8 x i16>, <8 x i16>* %p20, align 16
  %21 = load <8 x i16>, <8 x i16>* %p21, align 16
  %22 = load <8 x i16>, <8 x i16>* %p22, align 16
  %23 = load <8 x i16>, <8 x i16>* %p23, align 16
  %24 = load <8 x i16>, <8 x i16>* %p24, align 16
  %25 = load <8 x i16>, <8 x i16>* %p25, align 16
  %26 = load <8 x i16>, <8 x i16>* %p26, align 16
  %27 = load <8 x i16>, <8 x i16>* %p27, align 16
  %28 = load <8 x i16>, <8 x i16>* %p28, align 16
  %29 = load <8 x i16>, <8 x i16>* %p29, align 16
  %30 = load <8 x i16>, <8 x i16>* %p30, align 16
  %31 = load <8 x i16>, <8 x i16>* %p31, align 16
  %32 = load <8 x i16>, <8 x i16>* %p32, align 16
  %33 = load <8 x i16>, <8 x i16>* %p33, align 16
  %r1  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0,   <8 x i16> %1)
  %r2  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r1,  <8 x i16> %2)
  %r3  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r2,  <8 x i16> %3)
  %r4  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r3,  <8 x i16> %4)
  %r5  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r4,  <8 x i16> %5)
  %r6  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r5,  <8 x i16> %6)
  %r7  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r6,  <8 x i16> %7)
  %r8  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r7,  <8 x i16> %8)
  %r9  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r8,  <8 x i16> %9)
  %r10 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r9,  <8 x i16> %10)
  %r11 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r10, <8 x i16> %11)
  %r12 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r11, <8 x i16> %12)
  %r13 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r12, <8 x i16> %13)
  %r14 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r13, <8 x i16> %14)
  %r15 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r14, <8 x i16> %15)
  %r16 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r15, <8 x i16> %16)
  %r17 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r16, <8 x i16> %17)
  %r18 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r17, <8 x i16> %18)
  %r19 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r18, <8 x i16> %19)
  %r20 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r19, <8 x i16> %20)
  %r21 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r20, <8 x i16> %21)
  %r22 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r21, <8 x i16> %22)
  %r23 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r22, <8 x i16> %23)
  %r24 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r23, <8 x i16> %24)
  %r25 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r24, <8 x i16> %25)
  %r26 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r25, <8 x i16> %26)
  %r27 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r26, <8 x i16> %27)
  %r28 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r27, <8 x i16> %28)
  %r29 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r28, <8 x i16> %29)
  %r30 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r29, <8 x i16> %30)
  %r31 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r30, <8 x i16> %31)
  %r32 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r31, <8 x i16> %32)
  %r33 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r32, <8 x i16> %33)
  %rx1  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %r33,   <8 x i16> %1)
  %rx2  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx1,  <8 x i16> %2)
  %rx3  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx2,  <8 x i16> %3)
  %rx4  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx3,  <8 x i16> %4)
  %rx5  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx4,  <8 x i16> %5)
  %rx6  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx5,  <8 x i16> %6)
  %rx7  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx6,  <8 x i16> %7)
  %rx8  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx7,  <8 x i16> %8)
  %rx9  = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx8,  <8 x i16> %9)
  %rx10 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx9,  <8 x i16> %10)
  %rx11 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx10, <8 x i16> %11)
  %rx12 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx11, <8 x i16> %12)
  %rx13 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx12, <8 x i16> %13)
  %rx14 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx13, <8 x i16> %14)
  %rx15 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx14, <8 x i16> %15)
  %rx16 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx15, <8 x i16> %16)
  %rx17 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx16, <8 x i16> %17)
  %rx18 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx17, <8 x i16> %18)
  %rx19 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx18, <8 x i16> %19)
  %rx20 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx19, <8 x i16> %20)
  %rx21 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx20, <8 x i16> %21)
  %rx22 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx21, <8 x i16> %22)
  %rx23 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx22, <8 x i16> %23)
  %rx24 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx23, <8 x i16> %24)
  %rx25 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx24, <8 x i16> %25)
  %rx26 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx25, <8 x i16> %26)
  %rx27 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx26, <8 x i16> %27)
  %rx28 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx27, <8 x i16> %28)
  %rx29 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx28, <8 x i16> %29)
  %rx30 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx29, <8 x i16> %30)
  %rx31 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx30, <8 x i16> %31)
  %rx32 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx31, <8 x i16> %32)
  %rx33 = call <8 x i16> @llvm.mips.addv.h(<8 x i16> %rx32, <8 x i16> %33)
  %res = call i32 @llvm.mips.copy.s.h(<8 x i16> %rx33, i32 0)
  ret i32 %res
}

declare <8 x i16> @llvm.mips.addv.h(<8 x i16>, <8 x i16>) nounwind
declare i32       @llvm.mips.copy.s.h(<8 x i16>, i32) nounwind

; CHECK: test_i16:
; CHECK: st.h {{.*}} Spill
; CHECK: st.h {{.*}} Spill
; CHECK: ld.h {{.*}} Reload
; CHECK: ld.h {{.*}} Reload
; CHECK: .size

define i32 @test_i32(<4 x i32>* %p0, <4 x i32>* %q1) nounwind {
entry:
  %p1  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 1
  %p2  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 2
  %p3  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 3
  %p4  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 4
  %p5  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 5
  %p6  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 6
  %p7  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 7
  %p8  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 8
  %p9  = getelementptr <4 x i32>, <4 x i32>* %p0, i32 9
  %p10 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 10
  %p11 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 11
  %p12 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 12
  %p13 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 13
  %p14 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 14
  %p15 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 15
  %p16 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 16
  %p17 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 17
  %p18 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 18
  %p19 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 19
  %p20 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 20
  %p21 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 21
  %p22 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 22
  %p23 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 23
  %p24 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 24
  %p25 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 25
  %p26 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 26
  %p27 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 27
  %p28 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 28
  %p29 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 29
  %p30 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 30
  %p31 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 31
  %p32 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 32
  %p33 = getelementptr <4 x i32>, <4 x i32>* %p0, i32 33
  %0  = load <4 x i32>, <4 x i32>* %p0, align 16
  %1  = load <4 x i32>, <4 x i32>* %p1, align 16
  %2  = load <4 x i32>, <4 x i32>* %p2, align 16
  %3  = load <4 x i32>, <4 x i32>* %p3, align 16
  %4  = load <4 x i32>, <4 x i32>* %p4, align 16
  %5  = load <4 x i32>, <4 x i32>* %p5, align 16
  %6  = load <4 x i32>, <4 x i32>* %p6, align 16
  %7  = load <4 x i32>, <4 x i32>* %p7, align 16
  %8  = load <4 x i32>, <4 x i32>* %p8, align 16
  %9  = load <4 x i32>, <4 x i32>* %p9, align 16
  %10 = load <4 x i32>, <4 x i32>* %p10, align 16
  %11 = load <4 x i32>, <4 x i32>* %p11, align 16
  %12 = load <4 x i32>, <4 x i32>* %p12, align 16
  %13 = load <4 x i32>, <4 x i32>* %p13, align 16
  %14 = load <4 x i32>, <4 x i32>* %p14, align 16
  %15 = load <4 x i32>, <4 x i32>* %p15, align 16
  %16 = load <4 x i32>, <4 x i32>* %p16, align 16
  %17 = load <4 x i32>, <4 x i32>* %p17, align 16
  %18 = load <4 x i32>, <4 x i32>* %p18, align 16
  %19 = load <4 x i32>, <4 x i32>* %p19, align 16
  %20 = load <4 x i32>, <4 x i32>* %p20, align 16
  %21 = load <4 x i32>, <4 x i32>* %p21, align 16
  %22 = load <4 x i32>, <4 x i32>* %p22, align 16
  %23 = load <4 x i32>, <4 x i32>* %p23, align 16
  %24 = load <4 x i32>, <4 x i32>* %p24, align 16
  %25 = load <4 x i32>, <4 x i32>* %p25, align 16
  %26 = load <4 x i32>, <4 x i32>* %p26, align 16
  %27 = load <4 x i32>, <4 x i32>* %p27, align 16
  %28 = load <4 x i32>, <4 x i32>* %p28, align 16
  %29 = load <4 x i32>, <4 x i32>* %p29, align 16
  %30 = load <4 x i32>, <4 x i32>* %p30, align 16
  %31 = load <4 x i32>, <4 x i32>* %p31, align 16
  %32 = load <4 x i32>, <4 x i32>* %p32, align 16
  %33 = load <4 x i32>, <4 x i32>* %p33, align 16
  %r1 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %1)
  %r2 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r1, <4 x i32> %2)
  %r3 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r2, <4 x i32> %3)
  %r4 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r3, <4 x i32> %4)
  %r5 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r4, <4 x i32> %5)
  %r6 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r5, <4 x i32> %6)
  %r7 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r6, <4 x i32> %7)
  %r8 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r7, <4 x i32> %8)
  %r9 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r8, <4 x i32> %9)
  %r10 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r9, <4 x i32> %10)
  %r11 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r10, <4 x i32> %11)
  %r12 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r11, <4 x i32> %12)
  %r13 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r12, <4 x i32> %13)
  %r14 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r13, <4 x i32> %14)
  %r15 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r14, <4 x i32> %15)
  %r16 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r15, <4 x i32> %16)
  %r17 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r16, <4 x i32> %17)
  %r18 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r17, <4 x i32> %18)
  %r19 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r18, <4 x i32> %19)
  %r20 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r19, <4 x i32> %20)
  %r21 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r20, <4 x i32> %21)
  %r22 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r21, <4 x i32> %22)
  %r23 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r22, <4 x i32> %23)
  %r24 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r23, <4 x i32> %24)
  %r25 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r24, <4 x i32> %25)
  %r26 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r25, <4 x i32> %26)
  %r27 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r26, <4 x i32> %27)
  %r28 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r27, <4 x i32> %28)
  %r29 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r28, <4 x i32> %29)
  %r30 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r29, <4 x i32> %30)
  %r31 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r30, <4 x i32> %31)
  %r32 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r31, <4 x i32> %32)
  %r33 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r32, <4 x i32> %33)
  %rx1 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %r33, <4 x i32> %1)
  %rx2 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx1, <4 x i32> %2)
  %rx3 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx2, <4 x i32> %3)
  %rx4 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx3, <4 x i32> %4)
  %rx5 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx4, <4 x i32> %5)
  %rx6 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx5, <4 x i32> %6)
  %rx7 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx6, <4 x i32> %7)
  %rx8 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx7, <4 x i32> %8)
  %rx9 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx8, <4 x i32> %9)
  %rx10 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx9, <4 x i32> %10)
  %rx11 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx10, <4 x i32> %11)
  %rx12 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx11, <4 x i32> %12)
  %rx13 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx12, <4 x i32> %13)
  %rx14 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx13, <4 x i32> %14)
  %rx15 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx14, <4 x i32> %15)
  %rx16 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx15, <4 x i32> %16)
  %rx17 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx16, <4 x i32> %17)
  %rx18 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx17, <4 x i32> %18)
  %rx19 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx18, <4 x i32> %19)
  %rx20 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx19, <4 x i32> %20)
  %rx21 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx20, <4 x i32> %21)
  %rx22 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx21, <4 x i32> %22)
  %rx23 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx22, <4 x i32> %23)
  %rx24 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx23, <4 x i32> %24)
  %rx25 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx24, <4 x i32> %25)
  %rx26 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx25, <4 x i32> %26)
  %rx27 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx26, <4 x i32> %27)
  %rx28 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx27, <4 x i32> %28)
  %rx29 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx28, <4 x i32> %29)
  %rx30 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx29, <4 x i32> %30)
  %rx31 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx30, <4 x i32> %31)
  %rx32 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx31, <4 x i32> %32)
  %rx33 = call <4 x i32> @llvm.mips.addv.w(<4 x i32> %rx32, <4 x i32> %33)
  %res = call i32 @llvm.mips.copy.s.w(<4 x i32> %rx33, i32 0)
  ret i32 %res
}

declare <4 x i32> @llvm.mips.addv.w(<4 x i32>, <4 x i32>) nounwind
declare i32       @llvm.mips.copy.s.w(<4 x i32>, i32) nounwind

; CHECK: test_i32:
; CHECK: st.w {{.*}} Spill
; CHECK: st.w {{.*}} Spill
; CHECK: ld.w {{.*}} Reload
; CHECK: ld.w {{.*}} Reload
; CHECK: .size

define i32 @test_i64(<2 x i64>* %p0, <2 x i64>* %q1) nounwind {
entry:
  %p1  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 1
  %p2  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 2
  %p3  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 3
  %p4  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 4
  %p5  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 5
  %p6  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 6
  %p7  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 7
  %p8  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 8
  %p9  = getelementptr <2 x i64>, <2 x i64>* %p0, i32 9
  %p10 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 10
  %p11 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 11
  %p12 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 12
  %p13 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 13
  %p14 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 14
  %p15 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 15
  %p16 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 16
  %p17 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 17
  %p18 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 18
  %p19 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 19
  %p20 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 20
  %p21 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 21
  %p22 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 22
  %p23 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 23
  %p24 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 24
  %p25 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 25
  %p26 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 26
  %p27 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 27
  %p28 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 28
  %p29 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 29
  %p30 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 30
  %p31 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 31
  %p32 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 32
  %p33 = getelementptr <2 x i64>, <2 x i64>* %p0, i32 33
  %0  = load <2 x i64>, <2 x i64>* %p0, align 16
  %1  = load <2 x i64>, <2 x i64>* %p1, align 16
  %2  = load <2 x i64>, <2 x i64>* %p2, align 16
  %3  = load <2 x i64>, <2 x i64>* %p3, align 16
  %4  = load <2 x i64>, <2 x i64>* %p4, align 16
  %5  = load <2 x i64>, <2 x i64>* %p5, align 16
  %6  = load <2 x i64>, <2 x i64>* %p6, align 16
  %7  = load <2 x i64>, <2 x i64>* %p7, align 16
  %8  = load <2 x i64>, <2 x i64>* %p8, align 16
  %9  = load <2 x i64>, <2 x i64>* %p9, align 16
  %10 = load <2 x i64>, <2 x i64>* %p10, align 16
  %11 = load <2 x i64>, <2 x i64>* %p11, align 16
  %12 = load <2 x i64>, <2 x i64>* %p12, align 16
  %13 = load <2 x i64>, <2 x i64>* %p13, align 16
  %14 = load <2 x i64>, <2 x i64>* %p14, align 16
  %15 = load <2 x i64>, <2 x i64>* %p15, align 16
  %16 = load <2 x i64>, <2 x i64>* %p16, align 16
  %17 = load <2 x i64>, <2 x i64>* %p17, align 16
  %18 = load <2 x i64>, <2 x i64>* %p18, align 16
  %19 = load <2 x i64>, <2 x i64>* %p19, align 16
  %20 = load <2 x i64>, <2 x i64>* %p20, align 16
  %21 = load <2 x i64>, <2 x i64>* %p21, align 16
  %22 = load <2 x i64>, <2 x i64>* %p22, align 16
  %23 = load <2 x i64>, <2 x i64>* %p23, align 16
  %24 = load <2 x i64>, <2 x i64>* %p24, align 16
  %25 = load <2 x i64>, <2 x i64>* %p25, align 16
  %26 = load <2 x i64>, <2 x i64>* %p26, align 16
  %27 = load <2 x i64>, <2 x i64>* %p27, align 16
  %28 = load <2 x i64>, <2 x i64>* %p28, align 16
  %29 = load <2 x i64>, <2 x i64>* %p29, align 16
  %30 = load <2 x i64>, <2 x i64>* %p30, align 16
  %31 = load <2 x i64>, <2 x i64>* %p31, align 16
  %32 = load <2 x i64>, <2 x i64>* %p32, align 16
  %33 = load <2 x i64>, <2 x i64>* %p33, align 16
  %r1  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0,   <2 x i64> %1)
  %r2  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r1,  <2 x i64> %2)
  %r3  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r2,  <2 x i64> %3)
  %r4  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r3,  <2 x i64> %4)
  %r5  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r4,  <2 x i64> %5)
  %r6  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r5,  <2 x i64> %6)
  %r7  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r6,  <2 x i64> %7)
  %r8  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r7,  <2 x i64> %8)
  %r9  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r8,  <2 x i64> %9)
  %r10 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r9,  <2 x i64> %10)
  %r11 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r10, <2 x i64> %11)
  %r12 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r11, <2 x i64> %12)
  %r13 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r12, <2 x i64> %13)
  %r14 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r13, <2 x i64> %14)
  %r15 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r14, <2 x i64> %15)
  %r16 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r15, <2 x i64> %16)
  %r17 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r16, <2 x i64> %17)
  %r18 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r17, <2 x i64> %18)
  %r19 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r18, <2 x i64> %19)
  %r20 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r19, <2 x i64> %20)
  %r21 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r20, <2 x i64> %21)
  %r22 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r21, <2 x i64> %22)
  %r23 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r22, <2 x i64> %23)
  %r24 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r23, <2 x i64> %24)
  %r25 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r24, <2 x i64> %25)
  %r26 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r25, <2 x i64> %26)
  %r27 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r26, <2 x i64> %27)
  %r28 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r27, <2 x i64> %28)
  %r29 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r28, <2 x i64> %29)
  %r30 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r29, <2 x i64> %30)
  %r31 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r30, <2 x i64> %31)
  %r32 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r31, <2 x i64> %32)
  %r33 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r32, <2 x i64> %33)
  %rx1  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %r33,  <2 x i64> %1)
  %rx2  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx1,  <2 x i64> %2)
  %rx3  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx2,  <2 x i64> %3)
  %rx4  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx3,  <2 x i64> %4)
  %rx5  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx4,  <2 x i64> %5)
  %rx6  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx5,  <2 x i64> %6)
  %rx7  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx6,  <2 x i64> %7)
  %rx8  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx7,  <2 x i64> %8)
  %rx9  = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx8,  <2 x i64> %9)
  %rx10 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx9,  <2 x i64> %10)
  %rx11 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx10, <2 x i64> %11)
  %rx12 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx11, <2 x i64> %12)
  %rx13 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx12, <2 x i64> %13)
  %rx14 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx13, <2 x i64> %14)
  %rx15 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx14, <2 x i64> %15)
  %rx16 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx15, <2 x i64> %16)
  %rx17 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx16, <2 x i64> %17)
  %rx18 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx17, <2 x i64> %18)
  %rx19 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx18, <2 x i64> %19)
  %rx20 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx19, <2 x i64> %20)
  %rx21 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx20, <2 x i64> %21)
  %rx22 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx21, <2 x i64> %22)
  %rx23 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx22, <2 x i64> %23)
  %rx24 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx23, <2 x i64> %24)
  %rx25 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx24, <2 x i64> %25)
  %rx26 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx25, <2 x i64> %26)
  %rx27 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx26, <2 x i64> %27)
  %rx28 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx27, <2 x i64> %28)
  %rx29 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx28, <2 x i64> %29)
  %rx30 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx29, <2 x i64> %30)
  %rx31 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx30, <2 x i64> %31)
  %rx32 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx31, <2 x i64> %32)
  %rx33 = call <2 x i64> @llvm.mips.addv.d(<2 x i64> %rx32, <2 x i64> %33)
  %res1 = bitcast <2 x i64> %rx33 to <4 x i32>
  %res = call i32 @llvm.mips.copy.s.w(<4 x i32> %res1, i32 0)
  ret i32 %res
}

declare <2 x i64> @llvm.mips.addv.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: test_i64:
; CHECK: st.d {{.*}} Spill
; CHECK: st.d {{.*}} Spill
; CHECK: ld.d {{.*}} Reload
; CHECK: ld.d {{.*}} Reload
; CHECK: .size
