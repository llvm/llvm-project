; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

declare void @llvm.write_register.i32(metadata, i32) #1
declare void @llvm.write_register.i64(metadata, i64) #1
declare i32 @llvm.read_register.i32(metadata) #2
declare i64 @llvm.read_register.i64(metadata) #2

; CHECK-LABEL: reg_r0:
; CHECK: r0 = #1
define dso_local i32 @reg_r0() #0 {
entry:
  call void @llvm.write_register.i32(metadata !0, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %0
}

; CHECK-LABEL: reg_r1:
; CHECK: r1 = #1
; CHECK: r0 = r1
define dso_local i32 @reg_r1() #0 {
entry:
  call void @llvm.write_register.i32(metadata !1, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !1)
  ret i32 %0
}

; CHECK-LABEL: reg_r2:
; CHECK: r2 = #1
; CHECK: r0 = r2
define dso_local i32 @reg_r2() #0 {
entry:
  call void @llvm.write_register.i32(metadata !2, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !2)
  ret i32 %0
}

; CHECK-LABEL: reg_r3:
; CHECK: r3 = #1
; CHECK: r0 = r3
define dso_local i32 @reg_r3() #0 {
entry:
  call void @llvm.write_register.i32(metadata !3, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !3)
  ret i32 %0
}

; CHECK-LABEL: reg_r4:
; CHECK: r4 = #1
; CHECK: r0 = r4
define dso_local i32 @reg_r4() #0 {
entry:
  call void @llvm.write_register.i32(metadata !4, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !4)
  ret i32 %0
}

; CHECK-LABEL: reg_r5:
; CHECK: r5 = #1
; CHECK: r0 = r5
define dso_local i32 @reg_r5() #0 {
entry:
  call void @llvm.write_register.i32(metadata !5, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !5)
  ret i32 %0
}

; CHECK-LABEL: reg_r6:
; CHECK: r6 = #1
; CHECK: r0 = r6
define dso_local i32 @reg_r6() #0 {
entry:
  call void @llvm.write_register.i32(metadata !6, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !6)
  ret i32 %0
}

; CHECK-LABEL: reg_r7:
; CHECK: r7 = #1
; CHECK: r0 = r7
define dso_local i32 @reg_r7() #0 {
entry:
  call void @llvm.write_register.i32(metadata !7, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !7)
  ret i32 %0
}

; CHECK-LABEL: reg_r8:
; CHECK: r8 = #1
; CHECK: r0 = r8
define dso_local i32 @reg_r8() #0 {
entry:
  call void @llvm.write_register.i32(metadata !8, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !8)
  ret i32 %0
}

; CHECK-LABEL: reg_r9:
; CHECK: r9 = #1
; CHECK: r0 = r9
define dso_local i32 @reg_r9() #0 {
entry:
  call void @llvm.write_register.i32(metadata !9, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !9)
  ret i32 %0
}

; CHECK-LABEL: reg_r10:
; CHECK: r10 = #1
; CHECK: r0 = r10
define dso_local i32 @reg_r10() #0 {
entry:
  call void @llvm.write_register.i32(metadata !10, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !10)
  ret i32 %0
}

; CHECK-LABEL: reg_r11:
; CHECK: r11 = #1
; CHECK: r0 = r11
define dso_local i32 @reg_r11() #0 {
entry:
  call void @llvm.write_register.i32(metadata !11, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !11)
  ret i32 %0
}

; CHECK-LABEL: reg_r12:
; CHECK: r12 = #1
; CHECK: r0 = r12
define dso_local i32 @reg_r12() #0 {
entry:
  call void @llvm.write_register.i32(metadata !12, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !12)
  ret i32 %0
}

; CHECK-LABEL: reg_r13:
; CHECK: r13 = #1
; CHECK: r0 = r13
define dso_local i32 @reg_r13() #0 {
entry:
  call void @llvm.write_register.i32(metadata !13, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !13)
  ret i32 %0
}

; CHECK-LABEL: reg_r14:
; CHECK: r14 = #1
; CHECK: r0 = r14
define dso_local i32 @reg_r14() #0 {
entry:
  call void @llvm.write_register.i32(metadata !14, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !14)
  ret i32 %0
}

; CHECK-LABEL: reg_r15:
; CHECK: r15 = #1
; CHECK: r0 = r15
define dso_local i32 @reg_r15() #0 {
entry:
  call void @llvm.write_register.i32(metadata !15, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !15)
  ret i32 %0
}

; CHECK-LABEL: reg_r16:
; CHECK: r16 = #1
; CHECK: r0 = r16
define dso_local i32 @reg_r16() #0 {
entry:
  call void @llvm.write_register.i32(metadata !16, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !16)
  ret i32 %0
}

; CHECK-LABEL: reg_r17:
; CHECK: r17 = #1
; CHECK: r0 = r17
define dso_local i32 @reg_r17() #0 {
entry:
  call void @llvm.write_register.i32(metadata !17, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !17)
  ret i32 %0
}

; CHECK-LABEL: reg_r18:
; CHECK: r18 = #1
; CHECK: r0 = r18
define dso_local i32 @reg_r18() #0 {
entry:
  call void @llvm.write_register.i32(metadata !18, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !18)
  ret i32 %0
}

; CHECK-LABEL: reg_r19:
; CHECK: r19 = #1
; CHECK: r0 = r19
define dso_local i32 @reg_r19() #0 {
entry:
  call void @llvm.write_register.i32(metadata !19, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !19)
  ret i32 %0
}

; CHECK-LABEL: reg_r20:
; CHECK: r20 = #1
; CHECK: r0 = r20
define dso_local i32 @reg_r20() #0 {
entry:
  call void @llvm.write_register.i32(metadata !20, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !20)
  ret i32 %0
}

; CHECK-LABEL: reg_r21:
; CHECK: r21 = #1
; CHECK: r0 = r21
define dso_local i32 @reg_r21() #0 {
entry:
  call void @llvm.write_register.i32(metadata !21, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !21)
  ret i32 %0
}

; CHECK-LABEL: reg_r22:
; CHECK: r22 = #1
; CHECK: r0 = r22
define dso_local i32 @reg_r22() #0 {
entry:
  call void @llvm.write_register.i32(metadata !22, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !22)
  ret i32 %0
}

; CHECK-LABEL: reg_r23:
; CHECK: r23 = #1
; CHECK: r0 = r23
define dso_local i32 @reg_r23() #0 {
entry:
  call void @llvm.write_register.i32(metadata !23, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !23)
  ret i32 %0
}

; CHECK-LABEL: reg_r24:
; CHECK: r24 = #1
; CHECK: r0 = r24
define dso_local i32 @reg_r24() #0 {
entry:
  call void @llvm.write_register.i32(metadata !24, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !24)
  ret i32 %0
}

; CHECK-LABEL: reg_r25:
; CHECK: r25 = #1
; CHECK: r0 = r25
define dso_local i32 @reg_r25() #0 {
entry:
  call void @llvm.write_register.i32(metadata !25, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !25)
  ret i32 %0
}

; CHECK-LABEL: reg_r26:
; CHECK: r26 = #1
; CHECK: r0 = r26
define dso_local i32 @reg_r26() #0 {
entry:
  call void @llvm.write_register.i32(metadata !26, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !26)
  ret i32 %0
}

; CHECK-LABEL: reg_r27:
; CHECK: r27 = #1
; CHECK: r0 = r27
define dso_local i32 @reg_r27() #0 {
entry:
  call void @llvm.write_register.i32(metadata !27, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !27)
  ret i32 %0
}

; CHECK-LABEL: reg_r28:
; CHECK: r28 = #1
; CHECK: r0 = r28
define dso_local i32 @reg_r28() #0 {
entry:
  call void @llvm.write_register.i32(metadata !28, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !28)
  ret i32 %0
}

; CHECK-LABEL: reg_r29:
; CHECK: r29 = #1
; CHECK: r0 = r29
define dso_local i32 @reg_r29() #0 {
entry:
  call void @llvm.write_register.i32(metadata !29, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !29)
  ret i32 %0
}

; CHECK-LABEL: reg_r30:
; CHECK: r30 = #1
; CHECK: r0 = r30
define dso_local i32 @reg_r30() #0 {
entry:
  call void @llvm.write_register.i32(metadata !30, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !30)
  ret i32 %0
}

; CHECK-LABEL: reg_r31:
; CHECK: r31 = #1
; CHECK: r0 = r31
define dso_local i32 @reg_r31() #0 {
entry:
  call void @llvm.write_register.i32(metadata !31, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  ret i32 %0
}

; CHECK-LABEL: reg_d0:
; CHECK: r1:0 = combine(#0,#1)
define dso_local i64 @reg_d0() #0 {
entry:
  call void @llvm.write_register.i64(metadata !32, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !32)
  ret i64 %0
}

; CHECK-LABEL: reg_d1:
; CHECK: r3:2 = combine(#0,#1)
; CHECK: r1:0 = combine(r3,r2)
define dso_local i64 @reg_d1() #0 {
entry:
  call void @llvm.write_register.i64(metadata !33, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !33)
  ret i64 %0
}

; CHECK-LABEL: reg_d2:
; CHECK: r5:4 = combine(#0,#1)
; CHECK: r1:0 = combine(r5,r4)
define dso_local i64 @reg_d2() #0 {
entry:
  call void @llvm.write_register.i64(metadata !34, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !34)
  ret i64 %0
}

; CHECK-LABEL: reg_d3:
; CHECK: r7:6 = combine(#0,#1)
; CHECK: r1:0 = combine(r7,r6)
define dso_local i64 @reg_d3() #0 {
entry:
  call void @llvm.write_register.i64(metadata !35, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !35)
  ret i64 %0
}

; CHECK-LABEL: reg_d4:
; CHECK: r9:8 = combine(#0,#1)
; CHECK: r1:0 = combine(r9,r8)
define dso_local i64 @reg_d4() #0 {
entry:
  call void @llvm.write_register.i64(metadata !36, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !36)
  ret i64 %0
}

; CHECK-LABEL: reg_d5:
; CHECK: r11:10 = combine(#0,#1)
; CHECK: r1:0 = combine(r11,r10)
define dso_local i64 @reg_d5() #0 {
entry:
  call void @llvm.write_register.i64(metadata !37, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !37)
  ret i64 %0
}

; CHECK-LABEL: reg_d6:
; CHECK: r13:12 = combine(#0,#1)
; CHECK: r1:0 = combine(r13,r12)
define dso_local i64 @reg_d6() #0 {
entry:
  call void @llvm.write_register.i64(metadata !38, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !38)
  ret i64 %0
}

; CHECK-LABEL: reg_d7:
; CHECK: r15:14 = combine(#0,#1)
; CHECK: r1:0 = combine(r15,r14)
define dso_local i64 @reg_d7() #0 {
entry:
  call void @llvm.write_register.i64(metadata !39, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !39)
  ret i64 %0
}

; CHECK-LABEL: reg_d8:
; CHECK: r17:16 = combine(#0,#1)
; CHECK: r1:0 = combine(r17,r16)
define dso_local i64 @reg_d8() #0 {
entry:
  call void @llvm.write_register.i64(metadata !40, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !40)
  ret i64 %0
}

; CHECK-LABEL: reg_d9:
; CHECK: r19:18 = combine(#0,#1)
; CHECK: r1:0 = combine(r19,r18)
define dso_local i64 @reg_d9() #0 {
entry:
  call void @llvm.write_register.i64(metadata !41, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !41)
  ret i64 %0
}

; CHECK-LABEL: reg_d10:
; CHECK: r21:20 = combine(#0,#1)
; CHECK: r1:0 = combine(r21,r20)
define dso_local i64 @reg_d10() #0 {
entry:
  call void @llvm.write_register.i64(metadata !42, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !42)
  ret i64 %0
}

; CHECK-LABEL: reg_d11:
; CHECK: r23:22 = combine(#0,#1)
; CHECK: r1:0 = combine(r23,r22)
define dso_local i64 @reg_d11() #0 {
entry:
  call void @llvm.write_register.i64(metadata !43, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !43)
  ret i64 %0
}

; CHECK-LABEL: reg_d12:
; CHECK: r25:24 = combine(#0,#1)
; CHECK: r1:0 = combine(r25,r24)
define dso_local i64 @reg_d12() #0 {
entry:
  call void @llvm.write_register.i64(metadata !44, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !44)
  ret i64 %0
}

; CHECK-LABEL: reg_d13:
; CHECK: r27:26 = combine(#0,#1)
; CHECK: r1:0 = combine(r27,r26)
define dso_local i64 @reg_d13() #0 {
entry:
  call void @llvm.write_register.i64(metadata !45, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !45)
  ret i64 %0
}

; CHECK-LABEL: reg_d14:
; CHECK: r29:28 = combine(#0,#1)
; CHECK: r1:0 = combine(r29,r28)
define dso_local i64 @reg_d14() #0 {
entry:
  call void @llvm.write_register.i64(metadata !46, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !46)
  ret i64 %0
}

; CHECK-LABEL: reg_d15:
; CHECK: r31:30 = combine(#0,#1)
; CHECK: r1:0 = combine(r31,r30)
define dso_local i64 @reg_d15() #0 {
entry:
  call void @llvm.write_register.i64(metadata !47, i64 1)
  %0 = call i64 @llvm.read_register.i64(metadata !47)
  ret i64 %0
}

; CHECK-LABEL: reg_sp:
; CHECK: r29 = #1
; CHECK: r0 = r29
define dso_local i32 @reg_sp() #0 {
entry:
  call void @llvm.write_register.i32(metadata !48, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !48)
  ret i32 %0
}

; CHECK-LABEL: reg_fp:
; CHECK: r30 = #1
; CHECK: r0 = r30
define dso_local i32 @reg_fp() #0 {
entry:
  call void @llvm.write_register.i32(metadata !49, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !49)
  ret i32 %0
}

; CHECK-LABEL: reg_lr:
; CHECK: r31 = #1
; CHECK: r0 = r31
define dso_local i32 @reg_lr() #0 {
entry:
  call void @llvm.write_register.i32(metadata !50, i32 1)
  %0 = call i32 @llvm.read_register.i32(metadata !50)
  ret i32 %0
}

; CHECK-LABEL: reg_p0:
; CHECK: p0 = r31
; CHECK: r0 = p0
define dso_local i32 @reg_p0() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !51, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !51)
  ret i32 %1
}

; CHECK-LABEL: reg_p1:
; CHECK: p1 = r31
; CHECK: r0 = p1
define dso_local i32 @reg_p1() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !52, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !52)
  ret i32 %1
}

; CHECK-LABEL: reg_p2:
; CHECK: p2 = r31
; CHECK: r0 = p2
define dso_local i32 @reg_p2() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !53, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !53)
  ret i32 %1
}

; CHECK-LABEL: reg_p3:
; CHECK: p3 = r31
; CHECK: r0 = p3
define dso_local i32 @reg_p3() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !54, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !54)
  ret i32 %1
}

; CHECK-LABEL: reg_sa0:
; CHECK: sa0 = r31
; CHECK: r0 = sa0
define dso_local i32 @reg_sa0() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !55, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !55)
  ret i32 %1
}

; CHECK-LABEL: reg_lc0:
; CHECK: lc0 = r31
; CHECK: r0 = lc0
define dso_local i32 @reg_lc0() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !56, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !56)
  ret i32 %1
}

; CHECK-LABEL: reg_sa1:
; CHECK: sa1 = r31
; CHECK: r0 = sa1
define dso_local i32 @reg_sa1() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !57, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !57)
  ret i32 %1
}

; CHECK-LABEL: reg_lc1:
; CHECK: lc1 = r31
; CHECK: r0 = lc1
define dso_local i32 @reg_lc1() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !58, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !58)
  ret i32 %1
}

; CHECK-LABEL: reg_m0:
; CHECK: m0 = r31
; CHECK: r0 = m0
define dso_local i32 @reg_m0() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !59, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !59)
  ret i32 %1
}

; CHECK-LABEL: reg_m1:
; CHECK: m1 = r31
; CHECK: r0 = m1
define dso_local i32 @reg_m1() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !60, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !60)
  ret i32 %1
}

; CHECK-LABEL: reg_usr:
; CHECK: usr = r31
; CHECK: r0 = usr
define dso_local i32 @reg_usr() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !61, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !61)
  ret i32 %1
}

; CHECK-LABEL: reg_ugp:
; CHECK: ugp = r31
; CHECK: r0 = ugp
define dso_local i32 @reg_ugp() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !31)
  call void @llvm.write_register.i32(metadata !62, i32 %0)
  %1 = call i32 @llvm.read_register.i32(metadata !62)
  ret i32 %1
}

attributes #0 = { noinline nounwind optnone "target-cpu"="hexagonv62" }
attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }

!llvm.named.register.r0 = !{!0}
!llvm.named.register.r1 = !{!1}
!llvm.named.register.r2 = !{!2}
!llvm.named.register.r3 = !{!3}
!llvm.named.register.r4 = !{!4}
!llvm.named.register.r5 = !{!5}
!llvm.named.register.r6 = !{!6}
!llvm.named.register.r7 = !{!7}
!llvm.named.register.r8 = !{!8}
!llvm.named.register.r9 = !{!9}
!llvm.named.register.r10 = !{!10}
!llvm.named.register.r11 = !{!11}
!llvm.named.register.r12 = !{!12}
!llvm.named.register.r13 = !{!13}
!llvm.named.register.r14 = !{!14}
!llvm.named.register.r15 = !{!15}
!llvm.named.register.r16 = !{!16}
!llvm.named.register.r17 = !{!17}
!llvm.named.register.r18 = !{!18}
!llvm.named.register.r19 = !{!19}
!llvm.named.register.r20 = !{!20}
!llvm.named.register.r21 = !{!21}
!llvm.named.register.r22 = !{!22}
!llvm.named.register.r23 = !{!23}
!llvm.named.register.r24 = !{!24}
!llvm.named.register.r25 = !{!25}
!llvm.named.register.r26 = !{!26}
!llvm.named.register.r27 = !{!27}
!llvm.named.register.r28 = !{!28}
!llvm.named.register.r29 = !{!29}
!llvm.named.register.r30 = !{!30}
!llvm.named.register.r31 = !{!31}
!llvm.named.register.r1\3A0 = !{!32}
!llvm.named.register.r3\3A2 = !{!33}
!llvm.named.register.r5\3A4 = !{!34}
!llvm.named.register.r7\3A6 = !{!35}
!llvm.named.register.r9\3A8 = !{!36}
!llvm.named.register.r11\3A10 = !{!37}
!llvm.named.register.r13\3A12 = !{!38}
!llvm.named.register.r15\3A14 = !{!39}
!llvm.named.register.r17\3A16 = !{!40}
!llvm.named.register.r19\3A18 = !{!41}
!llvm.named.register.r21\3A20 = !{!42}
!llvm.named.register.r23\3A22 = !{!43}
!llvm.named.register.r25\3A24 = !{!44}
!llvm.named.register.r27\3A26 = !{!45}
!llvm.named.register.r29\3A28 = !{!46}
!llvm.named.register.r31\3A30 = !{!47}
!llvm.named.register.sp = !{!48}
!llvm.named.register.fp = !{!49}
!llvm.named.register.lr = !{!50}
!llvm.named.register.p0 = !{!51}
!llvm.named.register.p1 = !{!52}
!llvm.named.register.p2 = !{!53}
!llvm.named.register.p3 = !{!54}
!llvm.named.register.sa0 = !{!55}
!llvm.named.register.lc0 = !{!56}
!llvm.named.register.sa1 = !{!57}
!llvm.named.register.lc1 = !{!58}
!llvm.named.register.m0 = !{!59}
!llvm.named.register.m1 = !{!60}
!llvm.named.register.usr = !{!61}
!llvm.named.register.ugp = !{!62}

!0 = !{!"r0"}
!1 = !{!"r1"}
!2 = !{!"r2"}
!3 = !{!"r3"}
!4 = !{!"r4"}
!5 = !{!"r5"}
!6 = !{!"r6"}
!7 = !{!"r7"}
!8 = !{!"r8"}
!9 = !{!"r9"}
!10 = !{!"r10"}
!11 = !{!"r11"}
!12 = !{!"r12"}
!13 = !{!"r13"}
!14 = !{!"r14"}
!15 = !{!"r15"}
!16 = !{!"r16"}
!17 = !{!"r17"}
!18 = !{!"r18"}
!19 = !{!"r19"}
!20 = !{!"r20"}
!21 = !{!"r21"}
!22 = !{!"r22"}
!23 = !{!"r23"}
!24 = !{!"r24"}
!25 = !{!"r25"}
!26 = !{!"r26"}
!27 = !{!"r27"}
!28 = !{!"r28"}
!29 = !{!"r29"}
!30 = !{!"r30"}
!31 = !{!"r31"}
!32 = !{!"r1:0"}
!33 = !{!"r3:2"}
!34 = !{!"r5:4"}
!35 = !{!"r7:6"}
!36 = !{!"r9:8"}
!37 = !{!"r11:10"}
!38 = !{!"r13:12"}
!39 = !{!"r15:14"}
!40 = !{!"r17:16"}
!41 = !{!"r19:18"}
!42 = !{!"r21:20"}
!43 = !{!"r23:22"}
!44 = !{!"r25:24"}
!45 = !{!"r27:26"}
!46 = !{!"r29:28"}
!47 = !{!"r31:30"}
!48 = !{!"sp"}
!49 = !{!"fp"}
!50 = !{!"lr"}
!51 = !{!"p0"}
!52 = !{!"p1"}
!53 = !{!"p2"}
!54 = !{!"p3"}
!55 = !{!"sa0"}
!56 = !{!"lc0"}
!57 = !{!"sa1"}
!58 = !{!"lc1"}
!59 = !{!"m0"}
!60 = !{!"m1"}
!61 = !{!"usr"}
!62 = !{!"ugp"}
