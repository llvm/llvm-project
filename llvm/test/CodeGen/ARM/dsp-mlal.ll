; RUN: llc -mtriple=thumbv7m -mattr=+dsp %s -o - | FileCheck %s
; RUN: llc -mtriple=armv7a %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m -mattr=-dsp %s -o - | FileCheck --check-prefix=NODSP %s

define hidden i32 @SMMULR_SMMLAR(i32 %a, i32 %b0, i32 %b1, i32 %Xn, i32 %Xn1) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMULR_SMMLAR:
; CHECK: ldr r0, [sp]
; CHECK-NEXT: smmulr r0, {{(r0, r2|r2, r0)}}
; CHECK-NEXT: smmlar r0, {{(r1, r3|r3, r1)}}, r0
; NODSP-LABEL: SMMULR_SMMLAR:
; NODSP-NOT: smmulr
; NODSP-NOT: smmlar
  %conv = sext i32 %b1 to i64
  %conv1 = sext i32 %Xn1 to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, 2147483648
  %0 = and i64 %add, -4294967296
  %conv4 = sext i32 %b0 to i64
  %conv5 = sext i32 %Xn to i64
  %mul6 = mul nsw i64 %conv5, %conv4
  %add7 = add i64 %mul6, 2147483648
  %add8 = add i64 %add7, %0
  %1 = lshr i64 %add8, 32
  %conv10 = trunc i64 %1 to i32
  ret i32 %conv10
}

define hidden i32 @SMMULR(i32 %a, i32 %b) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMULR:
; CHECK: smmulr r0, {{(r0, r1|r1, r0)}}
; NODSP-LABEL: SMMULR:
; NODSP-NOT: smmulr
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, 2147483648
  %0 = lshr i64 %add, 32
  %conv2 = trunc i64 %0 to i32
  ret i32 %conv2
}

define hidden i32 @SMMUL(i32 %a, i32 %b) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMUL:
; CHECK: smmul r0, {{(r0, r1|r1, r0)}}
; NODSP-LABEL: SMMUL:
; NODSP-NOT: smmul
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %mul = mul nsw i64 %conv1, %conv
  %0 = lshr i64 %mul, 32
  %conv2 = trunc i64 %0 to i32
  ret i32 %conv2
}

define hidden i32 @SMMLSR(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMLSR:
; CHECK: smmlsr r0, {{(r1, r2|r2, r1)}}, r0
; NODSP-LABEL: SMMLSR:
; NODSP-NOT: smmlsr
  %conv6 = zext i32 %a to i64
  %shl = shl nuw i64 %conv6, 32
  %conv1 = sext i32 %b to i64
  %conv2 = sext i32 %c to i64
  %mul = mul nsw i64 %conv2, %conv1
  %sub = or i64 %shl, 2147483648
  %add = sub i64 %sub, %mul
  %0 = lshr i64 %add, 32
  %conv3 = trunc i64 %0 to i32
  ret i32 %conv3
}

define hidden i32 @NOT_SMMLSR(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: NOT_SMMLSR:
; CHECK-NOT: smmlsr
; NODSP-LABEL: NOT_SMMLSR:
; NODSP-NOT: smmlsr
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, 2147483648
  %0 = lshr i64 %add, 32
  %conv2 = trunc i64 %0 to i32
  %sub = sub nsw i32 %a, %conv2
  ret i32 %sub
}

define hidden i32 @SMMLS(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMLS:
; CHECK: smmls r0, {{(r1, r2|r2, r1)}}, r0
; NODSP-LABEL: SMMLS:
; NODSP-NOT: smmls
  %conv5 = zext i32 %a to i64
  %shl = shl nuw i64 %conv5, 32
  %conv1 = sext i32 %b to i64
  %conv2 = sext i32 %c to i64
  %mul = mul nsw i64 %conv2, %conv1
  %sub = sub nsw i64 %shl, %mul
  %0 = lshr i64 %sub, 32
  %conv3 = trunc i64 %0 to i32
  ret i32 %conv3
}

define hidden i32 @NOT_SMMLS(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: NOT_SMMLS:
; CHECK-NOT: smmls
; NODSP-LABEL: NOT_SMMLS:
; NODSP-NOT: smmls
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %0 = lshr i64 %mul, 32
  %conv2 = trunc i64 %0 to i32
  %sub = sub nsw i32 %a, %conv2
  ret i32 %sub
}

define hidden i32 @SMMLA(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMLA:
; CHECK: smmla r0, {{(r1, r2|r2, r1)}}, r0
; NODSP-LABEL: SMMLA:
; NODSP-NOT: smmla
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %0 = lshr i64 %mul, 32
  %conv2 = trunc i64 %0 to i32
  %add = add nsw i32 %conv2, %a
  ret i32 %add
}

define hidden i32 @SMMLAR(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: SMMLAR:
; CHECK: smmlar r0, {{(r1, r2|r2, r1)}}, r0
; NODSP-LABEL: SMMLAR:
; NODSP-NOT: smmlar
  %conv7 = zext i32 %a to i64
  %shl = shl nuw i64 %conv7, 32
  %conv1 = sext i32 %b to i64
  %conv2 = sext i32 %c to i64
  %mul = mul nsw i64 %conv2, %conv1
  %add = or i64 %shl, 2147483648
  %add3 = add i64 %add, %mul
  %0 = lshr i64 %add3, 32
  %conv4 = trunc i64 %0 to i32
  ret i32 %conv4
}

define hidden i32 @NOT_SMMLA(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
; CHECK-LABEL: NOT_SMMLA:
; CHECK-NOT: smmla
; NODSP-LABEL: NOT_SMMLA:
; NODSP-NOT: smmla
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %0 = lshr i64 %mul, 32
  %conv2 = trunc i64 %0 to i32
  %add = xor i32 %conv2, -2147483648
  %add3 = add i32 %add, %a
  ret i32 %add3
}
