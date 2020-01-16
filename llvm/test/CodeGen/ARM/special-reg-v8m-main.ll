; RUN: not llc < %s -mtriple=thumbv8m.base-none-eabi 2>&1 | FileCheck %s --check-prefix=BASELINE
; RUN: llc < %s -mtriple=thumbv8m.main-none-eabi -mattr=+dsp 2>&1 | FileCheck %s --check-prefix=MAINLINE

; BASELINE: LLVM ERROR: Invalid register name "faultmask_ns".

define i32 @read_mclass_registers() nounwind {
entry:
  ; MAINLINE-LABEL: read_mclass_registers:
  ; MAINLINE:   mrs r0, apsr
  ; MAINLINE:   mrs r1, iapsr
  ; MAINLINE:   mrs r1, eapsr
  ; MAINLINE:   mrs r1, xpsr
  ; MAINLINE:   mrs r1, ipsr
  ; MAINLINE:   mrs r1, epsr
  ; MAINLINE:   mrs r1, iepsr
  ; MAINLINE:   mrs r1, msp
  ; MAINLINE:   mrs r1, psp
  ; MAINLINE:   mrs r1, primask
  ; MAINLINE:   mrs r1, basepri
  ; MAINLINE:   mrs r1, basepri_max
  ; MAINLINE:   mrs r1, faultmask
  ; MAINLINE:   mrs r1, control
  ; MAINLINE:   mrs r1, msplim
  ; MAINLINE:   mrs r1, psplim
  ; MAINLINE:   mrs r1, msp_ns
  ; MAINLINE:   mrs r1, psp_ns
  ; MAINLINE:   mrs r1, msplim_ns
  ; MAINLINE:   mrs r1, psplim_ns
  ; MAINLINE:   mrs r1, primask_ns
  ; MAINLINE:   mrs r1, basepri_ns
  ; MAINLINE:   mrs r1, faultmask_ns
  ; MAINLINE:   mrs r1, control_ns
  ; MAINLINE:   mrs r1, sp_ns

  %0 = call i32 @llvm.read_register.i32(metadata !0)
  %1 = call i32 @llvm.read_register.i32(metadata !4)
  %add1 = add i32 %1, %0
  %2 = call i32 @llvm.read_register.i32(metadata !8)
  %add2 = add i32 %add1, %2
  %3 = call i32 @llvm.read_register.i32(metadata !12)
  %add3 = add i32 %add2, %3
  %4 = call i32 @llvm.read_register.i32(metadata !16)
  %add4 = add i32 %add3, %4
  %5 = call i32 @llvm.read_register.i32(metadata !17)
  %add5 = add i32 %add4, %5
  %6 = call i32 @llvm.read_register.i32(metadata !18)
  %add6 = add i32 %add5, %6
  %7 = call i32 @llvm.read_register.i32(metadata !19)
  %add7 = add i32 %add6, %7
  %8 = call i32 @llvm.read_register.i32(metadata !20)
  %add8 = add i32 %add7, %8
  %9 = call i32 @llvm.read_register.i32(metadata !21)
  %add9 = add i32 %add8, %9
  %10 = call i32 @llvm.read_register.i32(metadata !22)
  %add10 = add i32 %add9, %10
  %11 = call i32 @llvm.read_register.i32(metadata !23)
  %add11 = add i32 %add10, %11
  %12 = call i32 @llvm.read_register.i32(metadata !24)
  %add12 = add i32 %add11, %12
  %13 = call i32 @llvm.read_register.i32(metadata !25)
  %add13 = add i32 %add12, %13
  %14 = call i32 @llvm.read_register.i32(metadata !26)
  %add14 = add i32 %add13, %14
  %15 = call i32 @llvm.read_register.i32(metadata !27)
  %add15 = add i32 %add14, %15
  %16 = call i32 @llvm.read_register.i32(metadata !28)
  %add16 = add i32 %add15, %16
  %17 = call i32 @llvm.read_register.i32(metadata !29)
  %add17 = add i32 %add16, %17
  %18 = call i32 @llvm.read_register.i32(metadata !30)
  %add18 = add i32 %add17, %18
  %19 = call i32 @llvm.read_register.i32(metadata !31)
  %add19 = add i32 %add18, %19
  %20 = call i32 @llvm.read_register.i32(metadata !32)
  %add20 = add i32 %add19, %20
  %21 = call i32 @llvm.read_register.i32(metadata !33)
  %add21 = add i32 %add20, %21
  %22 = call i32 @llvm.read_register.i32(metadata !34)
  %add22 = add i32 %add21, %22
  %23 = call i32 @llvm.read_register.i32(metadata !35)
  %add23 = add i32 %add22, %23
  %24 = call i32 @llvm.read_register.i32(metadata !36)
  %add24 = add i32 %add23, %24
  ret i32 %add24
}

define void @write_mclass_registers(i32 %x) nounwind {
entry:
  ; MAINLINE-LABEL: write_mclass_registers:
  ; MAINLINE:   msr apsr_nzcvq, r0
  ; MAINLINE:   msr apsr_nzcvq, r0
  ; MAINLINE:   msr apsr_g, r0
  ; MAINLINE:   msr apsr_nzcvqg, r0
  ; MAINLINE:   msr iapsr_nzcvq, r0
  ; MAINLINE:   msr iapsr_nzcvq, r0
  ; MAINLINE:   msr iapsr_g, r0
  ; MAINLINE:   msr iapsr_nzcvqg, r0
  ; MAINLINE:   msr eapsr_nzcvq, r0
  ; MAINLINE:   msr eapsr_nzcvq, r0
  ; MAINLINE:   msr eapsr_g, r0
  ; MAINLINE:   msr eapsr_nzcvqg, r0
  ; MAINLINE:   msr xpsr_nzcvq, r0
  ; MAINLINE:   msr xpsr_nzcvq, r0
  ; MAINLINE:   msr xpsr_g, r0
  ; MAINLINE:   msr xpsr_nzcvqg, r0
  ; MAINLINE:   msr ipsr, r0
  ; MAINLINE:   msr epsr, r0
  ; MAINLINE:   msr iepsr, r0
  ; MAINLINE:   msr msp, r0
  ; MAINLINE:   msr psp, r0
  ; MAINLINE:   msr primask, r0
  ; MAINLINE:   msr basepri, r0
  ; MAINLINE:   msr basepri_max, r0
  ; MAINLINE:   msr faultmask, r0
  ; MAINLINE:   msr control, r0
  ; MAINLINE:   msr msplim, r0
  ; MAINLINE:   msr psplim, r0
  ; MAINLINE:   msr msp_ns, r0
  ; MAINLINE:   msr psp_ns, r0
  ; MAINLINE:   msr msplim_ns, r0
  ; MAINLINE:   msr psplim_ns, r0
  ; MAINLINE:   msr primask_ns, r0
  ; MAINLINE:   msr basepri_ns, r0
  ; MAINLINE:   msr faultmask_ns, r0
  ; MAINLINE:   msr control_ns, r0
  ; MAINLINE:   msr sp_ns, r0

  call void @llvm.write_register.i32(metadata !0, i32 %x)
  call void @llvm.write_register.i32(metadata !1, i32 %x)
  call void @llvm.write_register.i32(metadata !2, i32 %x)
  call void @llvm.write_register.i32(metadata !3, i32 %x)
  call void @llvm.write_register.i32(metadata !4, i32 %x)
  call void @llvm.write_register.i32(metadata !5, i32 %x)
  call void @llvm.write_register.i32(metadata !6, i32 %x)
  call void @llvm.write_register.i32(metadata !7, i32 %x)
  call void @llvm.write_register.i32(metadata !8, i32 %x)
  call void @llvm.write_register.i32(metadata !9, i32 %x)
  call void @llvm.write_register.i32(metadata !10, i32 %x)
  call void @llvm.write_register.i32(metadata !11, i32 %x)
  call void @llvm.write_register.i32(metadata !12, i32 %x)
  call void @llvm.write_register.i32(metadata !13, i32 %x)
  call void @llvm.write_register.i32(metadata !14, i32 %x)
  call void @llvm.write_register.i32(metadata !15, i32 %x)
  call void @llvm.write_register.i32(metadata !16, i32 %x)
  call void @llvm.write_register.i32(metadata !17, i32 %x)
  call void @llvm.write_register.i32(metadata !18, i32 %x)
  call void @llvm.write_register.i32(metadata !19, i32 %x)
  call void @llvm.write_register.i32(metadata !20, i32 %x)
  call void @llvm.write_register.i32(metadata !21, i32 %x)
  call void @llvm.write_register.i32(metadata !22, i32 %x)
  call void @llvm.write_register.i32(metadata !23, i32 %x)
  call void @llvm.write_register.i32(metadata !24, i32 %x)
  call void @llvm.write_register.i32(metadata !25, i32 %x)
  call void @llvm.write_register.i32(metadata !26, i32 %x)
  call void @llvm.write_register.i32(metadata !27, i32 %x)
  call void @llvm.write_register.i32(metadata !28, i32 %x)
  call void @llvm.write_register.i32(metadata !29, i32 %x)
  call void @llvm.write_register.i32(metadata !30, i32 %x)
  call void @llvm.write_register.i32(metadata !31, i32 %x)
  call void @llvm.write_register.i32(metadata !32, i32 %x)
  call void @llvm.write_register.i32(metadata !33, i32 %x)
  call void @llvm.write_register.i32(metadata !34, i32 %x)
  call void @llvm.write_register.i32(metadata !35, i32 %x)
  call void @llvm.write_register.i32(metadata !36, i32 %x)
  ret void
}

declare i32 @llvm.read_register.i32(metadata) nounwind
declare void @llvm.write_register.i32(metadata, i32) nounwind

!0 = !{!"apsr"}
!1 = !{!"apsr_nzcvq"}
!2 = !{!"apsr_g"}
!3 = !{!"apsr_nzcvqg"}
!4 = !{!"iapsr"}
!5 = !{!"iapsr_nzcvq"}
!6 = !{!"iapsr_g"}
!7 = !{!"iapsr_nzcvqg"}
!8 = !{!"eapsr"}
!9 = !{!"eapsr_nzcvq"}
!10 = !{!"eapsr_g"}
!11 = !{!"eapsr_nzcvqg"}
!12 = !{!"xpsr"}
!13 = !{!"xpsr_nzcvq"}
!14 = !{!"xpsr_g"}
!15 = !{!"xpsr_nzcvqg"}
!16 = !{!"ipsr"}
!17 = !{!"epsr"}
!18 = !{!"iepsr"}
!19 = !{!"msp"}
!20 = !{!"psp"}
!21 = !{!"primask"}
!22 = !{!"basepri"}
!23 = !{!"basepri_max"}
!24 = !{!"faultmask"}
!25 = !{!"control"}
!26 = !{!"msplim"}
!27 = !{!"psplim"}
!28 = !{!"msp_ns"}
!29 = !{!"psp_ns"}
!30 = !{!"msplim_ns"}
!31 = !{!"psplim_ns"}
!32 = !{!"primask_ns"}
!33 = !{!"basepri_ns"}
!34 = !{!"faultmask_ns"}
!35 = !{!"control_ns"}
!36 = !{!"sp_ns"}

