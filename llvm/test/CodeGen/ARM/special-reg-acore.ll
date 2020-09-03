; RUN: llc < %s -mtriple=arm-none-eabi -mcpu=cortex-a8 2>&1 | FileCheck %s --check-prefix=ACORE
; RUN: not --crash llc < %s -mtriple=thumb-none-eabi -mcpu=cortex-m4 2>&1 | FileCheck %s --check-prefix=MCORE

; MCORE: LLVM ERROR: Invalid register name "cpsr".

define i32 @read_cpsr() nounwind {
  ; ACORE-LABEL: read_cpsr:
  ; ACORE: mrs r0, apsr
  %reg = call i32 @llvm.read_register.i32(metadata !1)
  ret i32 %reg
}

define i32 @read_aclass_registers() nounwind {
entry:
  ; ACORE-LABEL: read_aclass_registers:
  ; ACORE: mrs r0, apsr
  ; ACORE: mrs r1, spsr

  %0 = call i32 @llvm.read_register.i32(metadata !0)
  %1 = call i32 @llvm.read_register.i32(metadata !1)
  %add1 = add i32 %1, %0
  %2 = call i32 @llvm.read_register.i32(metadata !2)
  %add2 = add i32 %add1, %2
  ret i32 %add2
}

define void @write_aclass_registers(i32 %x) nounwind {
entry:
  ; ACORE-LABEL: write_aclass_registers:
  ; ACORE:   msr APSR_nzcvq, r0
  ; ACORE:   msr APSR_g, r0
  ; ACORE:   msr APSR_nzcvqg, r0
  ; ACORE:   msr CPSR_c, r0
  ; ACORE:   msr CPSR_x, r0
  ; ACORE:   msr APSR_g, r0
  ; ACORE:   msr APSR_nzcvq, r0
  ; ACORE:   msr CPSR_fsxc, r0
  ; ACORE:   msr SPSR_c, r0
  ; ACORE:   msr SPSR_x, r0
  ; ACORE:   msr SPSR_s, r0
  ; ACORE:   msr SPSR_f, r0
  ; ACORE:   msr SPSR_fsxc, r0

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
  ret void
}

declare i32 @llvm.read_register.i32(metadata) nounwind
declare void @llvm.write_register.i32(metadata, i32) nounwind

!0 = !{!"apsr"}
!1 = !{!"cpsr"}
!2 = !{!"spsr"}
!3 = !{!"apsr_nzcvq"}
!4 = !{!"apsr_g"}
!5 = !{!"apsr_nzcvqg"}
!6 = !{!"cpsr_c"}
!7 = !{!"cpsr_x"}
!8 = !{!"cpsr_s"}
!9 = !{!"cpsr_f"}
!10 = !{!"cpsr_cxsf"}
!11 = !{!"spsr_c"}
!12 = !{!"spsr_x"}
!13 = !{!"spsr_s"}
!14 = !{!"spsr_f"}
!15 = !{!"spsr_cxsf"}
