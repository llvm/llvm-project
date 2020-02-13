; RUN: not llc < %s -mtriple=thumb-none-eabi -mcpu=cortex-m4 2>&1 | FileCheck %s --check-prefix=V7M
; RUN: llc < %s -mtriple=thumbv8m.base-none-eabi 2>&1 | FileCheck %s

; V7M: LLVM ERROR: Invalid register name "sp_ns".

define i32 @read_mclass_registers() nounwind {
entry:
  ; CHECK-LABEL: read_mclass_registers:
  ; CHECK:   mrs r0, apsr
  ; CHECK:   mrs r1, iapsr
  ; CHECK:   mrs r1, eapsr
  ; CHECK:   mrs r1, xpsr
  ; CHECK:   mrs r1, ipsr
  ; CHECK:   mrs r1, epsr
  ; CHECK:   mrs r1, iepsr
  ; CHECK:   mrs r1, msp
  ; CHECK:   mrs r1, psp
  ; CHECK:   mrs r1, primask
  ; CHECK:   mrs r1, control
  ; CHECK:   mrs r1, msplim
  ; CHECK:   mrs r1, psplim
  ; CHECK:   mrs r1, msp_ns
  ; CHECK:   mrs r1, psp_ns
  ; CHECK:   mrs r1, primask_ns
  ; CHECK:   mrs r1, control_ns
  ; CHECK:   mrs r1, sp_ns

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
  %10 = call i32 @llvm.read_register.i32(metadata !25)
  %add10 = add i32 %add9, %10
  %11 = call i32 @llvm.read_register.i32(metadata !26)
  %add11 = add i32 %add10, %11
  %12 = call i32 @llvm.read_register.i32(metadata !27)
  %add12 = add i32 %add11, %12
  %13 = call i32 @llvm.read_register.i32(metadata !28)
  %add13 = add i32 %add12, %13
  %14 = call i32 @llvm.read_register.i32(metadata !29)
  %add14 = add i32 %add13, %14
  %15 = call i32 @llvm.read_register.i32(metadata !32)
  %add15 = add i32 %add14, %15
  %16 = call i32 @llvm.read_register.i32(metadata !35)
  %add16 = add i32 %add15, %16
  %17 = call i32 @llvm.read_register.i32(metadata !36)
  %add17 = add i32 %add16, %17
  ret i32 %add10
}

define void @write_mclass_registers(i32 %x) nounwind {
entry:
  ; CHECK-LABEL: write_mclass_registers:
  ; CHECK:   msr apsr, r0
  ; CHECK:   msr apsr, r0
  ; CHECK:   msr iapsr, r0
  ; CHECK:   msr iapsr, r0
  ; CHECK:   msr eapsr, r0
  ; CHECK:   msr eapsr, r0
  ; CHECK:   msr xpsr, r0
  ; CHECK:   msr xpsr, r0
  ; CHECK:   msr ipsr, r0
  ; CHECK:   msr epsr, r0
  ; CHECK:   msr iepsr, r0
  ; CHECK:   msr msp, r0
  ; CHECK:   msr psp, r0
  ; CHECK:   msr primask, r0
  ; CHECK:   msr control, r0
  ; CHECK:   msr msplim, r0
  ; CHECK:   msr psplim, r0
  ; CHECK:   msr msp_ns, r0
  ; CHECK:   msr psp_ns, r0
  ; CHECK:   msr primask_ns, r0
  ; CHECK:   msr control_ns, r0
  ; CHECK:   msr sp_ns, r0

  call void @llvm.write_register.i32(metadata !0, i32 %x)
  call void @llvm.write_register.i32(metadata !1, i32 %x)
  call void @llvm.write_register.i32(metadata !4, i32 %x)
  call void @llvm.write_register.i32(metadata !5, i32 %x)
  call void @llvm.write_register.i32(metadata !8, i32 %x)
  call void @llvm.write_register.i32(metadata !9, i32 %x)
  call void @llvm.write_register.i32(metadata !12, i32 %x)
  call void @llvm.write_register.i32(metadata !13, i32 %x)
  call void @llvm.write_register.i32(metadata !16, i32 %x)
  call void @llvm.write_register.i32(metadata !17, i32 %x)
  call void @llvm.write_register.i32(metadata !18, i32 %x)
  call void @llvm.write_register.i32(metadata !19, i32 %x)
  call void @llvm.write_register.i32(metadata !20, i32 %x)
  call void @llvm.write_register.i32(metadata !21, i32 %x)
  call void @llvm.write_register.i32(metadata !25, i32 %x)
  call void @llvm.write_register.i32(metadata !26, i32 %x)
  call void @llvm.write_register.i32(metadata !27, i32 %x)
  call void @llvm.write_register.i32(metadata !28, i32 %x)
  call void @llvm.write_register.i32(metadata !29, i32 %x)
  call void @llvm.write_register.i32(metadata !32, i32 %x)
  call void @llvm.write_register.i32(metadata !35, i32 %x)
  call void @llvm.write_register.i32(metadata !36, i32 %x)
  ret void
}

declare i32 @llvm.read_register.i32(metadata) nounwind
declare void @llvm.write_register.i32(metadata, i32) nounwind

!0 = !{!"apsr"}
!1 = !{!"apsr_nzcvq"}
!4 = !{!"iapsr"}
!5 = !{!"iapsr_nzcvq"}
!8 = !{!"eapsr"}
!9 = !{!"eapsr_nzcvq"}
!12 = !{!"xpsr"}
!13 = !{!"xpsr_nzcvq"}
!16 = !{!"ipsr"}
!17 = !{!"epsr"}
!18 = !{!"iepsr"}
!19 = !{!"msp"}
!20 = !{!"psp"}
!21 = !{!"primask"}
!25 = !{!"control"}
!26 = !{!"msplim"}
!27 = !{!"psplim"}
!28 = !{!"msp_ns"}
!29 = !{!"psp_ns"}
!32 = !{!"primask_ns"}
!35 = !{!"control_ns"}
!36 = !{!"sp_ns"}

