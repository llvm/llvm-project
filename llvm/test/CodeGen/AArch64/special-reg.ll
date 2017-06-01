; RUN: llc < %s -mtriple=aarch64-none-eabi -mcpu=cortex-a57 2>&1 | FileCheck %s

define i64 @read_encoded_register() nounwind {
entry:
; CHECK-LABEL: read_encoded_register:
; CHECK: mrs x0, S1_2_C3_C4_5
  %reg = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %reg
}

define i64 @read_daif() nounwind {
entry:
; CHECK-LABEL: read_daif:
; CHECK: mrs x0, DAIF
  %reg = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %reg
}

define void @write_encoded_register(i64 %x) nounwind {
entry:
; CHECK-LABEL: write_encoded_register:
; CHECK: msr S1_2_C3_C4_5, x0
  call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

define void @write_daif(i64 %x) nounwind {
entry:
; CHECK-LABEL: write_daif:
; CHECK: msr DAIF, x0
  call void @llvm.write_register.i64(metadata !1, i64 %x)
  ret void
}

define void @write_daifset() nounwind {
entry:
; CHECK-LABEL: write_daifset:
; CHECK: msr DAIFSet, #2
  call void @llvm.write_register.i64(metadata !2, i64 2)
  ret void
}

declare i64 @llvm.read_register.i64(metadata) nounwind
declare void @llvm.write_register.i64(metadata, i64) nounwind

!0 = !{!"1:2:3:4:5"}
!1 = !{!"daif"}
!2 = !{!"daifset"}
