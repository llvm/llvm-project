; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s --check-prefix=V4
; RUN: llc -mtriple=bpfel -mcpu=v3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NO-HMUL
; RUN: llc -mtriple=bpfel -mcpu=v4 -disable-hmul -verify-machineinstrs < %s | FileCheck %s --check-prefix=NO-HMUL

define i64 @uhmul_rr(i64 %a, i64 %b) {
; V4-LABEL: uhmul_rr:
; V4:       r0 = r1
; V4-NEXT:  r0 uh*= r2                              # encoding: [0x2f,0x20,0x01,0x00,0x00,0x00,0x00,0x00]
; V4-NEXT:  exit
; NO-HMUL-LABEL: uhmul_rr:
; NO-HMUL-NOT:   uh*=
; NO-HMUL-NOT:   sh*=
; NO-HMUL:       call __multi3
; NO-HMUL:       exit
  %awide = zext i64 %a to i128
  %bwide = zext i64 %b to i128
  %product = mul i128 %awide, %bwide
  %high = lshr i128 %product, 64
  %result = trunc i128 %high to i64
  ret i64 %result
}

define i64 @shmul_rr(i64 %a, i64 %b) {
; V4-LABEL: shmul_rr:
; V4:       r0 = r1
; V4-NEXT:  r0 sh*= r2                              # encoding: [0x2f,0x20,0x02,0x00,0x00,0x00,0x00,0x00]
; V4-NEXT:  exit
; NO-HMUL-LABEL: shmul_rr:
; NO-HMUL-NOT:   uh*=
; NO-HMUL-NOT:   sh*=
; NO-HMUL:       call __multi3
; NO-HMUL:       exit
  %awide = sext i64 %a to i128
  %bwide = sext i64 %b to i128
  %product = mul i128 %awide, %bwide
  %high = ashr i128 %product, 64
  %result = trunc i128 %high to i64
  ret i64 %result
}

define i64 @uhmul_ri(i64 %a) {
; V4-LABEL: uhmul_ri:
; V4:       r0 = r1
; V4-NEXT:  r0 uh*= -2                              # encoding: [0x27,0x00,0x01,0x00,0xfe,0xff,0xff,0xff]
; V4-NEXT:  exit
; NO-HMUL-LABEL: uhmul_ri:
; NO-HMUL-NOT:   uh*=
; NO-HMUL-NOT:   sh*=
; NO-HMUL:       call __multi3
; NO-HMUL:       exit
  %awide = zext i64 %a to i128
  %product = mul i128 %awide, 18446744073709551614
  %high = lshr i128 %product, 64
  %result = trunc i128 %high to i64
  ret i64 %result
}

define i64 @shmul_ri(i64 %a) {
; V4-LABEL: shmul_ri:
; V4:       r0 = r1
; V4-NEXT:  r0 sh*= -2                              # encoding: [0x27,0x00,0x02,0x00,0xfe,0xff,0xff,0xff]
; V4-NEXT:  exit
; NO-HMUL-LABEL: shmul_ri:
; NO-HMUL-NOT:   uh*=
; NO-HMUL-NOT:   sh*=
; NO-HMUL:       exit
  %awide = sext i64 %a to i128
  %product = mul i128 %awide, -2
  %high = ashr i128 %product, 64
  %result = trunc i128 %high to i64
  ret i64 %result
}

define i64 @mul_low(i64 %a, i64 %b) {
; V4-LABEL: mul_low:
; V4:       r0 = r1
; V4-NEXT:  r0 *= r2                                # encoding: [0x2f,0x20,0x00,0x00,0x00,0x00,0x00,0x00]
; V4-NEXT:  exit
; NO-HMUL-LABEL: mul_low:
; NO-HMUL:       r0 *= r2
  %product = mul i64 %a, %b
  ret i64 %product
}
