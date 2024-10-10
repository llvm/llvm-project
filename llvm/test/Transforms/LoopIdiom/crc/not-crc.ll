; RUN: opt -passes=loop-idiom < %s -S -debug -recognize-crc 2>&1 | FileCheck %s

; crc16 incorrect xor inside loop
; CHECK: loop-idiom CRCRegonize: Cannot verify check bit!
; CHECK: crc[0]^data[0]
; CHECK: crc[1]^1
define i16 @crc16_incorrect_xor(i8 %data, i16 %crc) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.036 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %crc.addr.035 = phi i16 [ %crc, %entry ], [ %crc.addr.2, %for.body ]
  %data.addr.034 = phi i8 [ %data, %entry ], [ %1, %for.body ]
  %0 = trunc i16 %crc.addr.035 to i8
  %and33 = xor i8 %0, 25
  %xor = and i8 %and33, 1
  %1 = lshr i8 %data.addr.034, 1
  %cmp10.not = icmp eq i8 %xor, 0
  %2 = lshr i16 %crc.addr.035, 1
  %3 = xor i16 %2, -24575
  %crc.addr.2 = select i1 %cmp10.not, i16 %2, i16 %3
  %inc = add nuw nsw i8 %i.036, 1
  %cmp = icmp ult i8 %inc, 8
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %crc.addr.0.lcssa = phi i16 [ %crc.addr.2, %for.body ]
  ret i16 %crc.addr.0.lcssa
}

; Two byte at a time crc not supported
; CHECK-NOT: loop-idiom CRCRegonize: This looks like crc!
define i16 @crc16_reversed_data16(i16 %data, i16 %crc) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.036 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %crc.addr.035 = phi i16 [ %crc, %entry ], [ %crc.addr.2, %for.body ]
  %data.addr.034 = phi i16 [ %data, %entry ], [ %0, %for.body ]
  %and33 = xor i16 %crc.addr.035, %data.addr.034
  %xor = and i16 %and33, 1
  %0 = lshr i16 %data.addr.034, 1
  %cmp10.not = icmp eq i16 %xor, 0
  %1 = lshr i16 %crc.addr.035, 1
  %2 = xor i16 %1, -24575
  %crc.addr.2 = select i1 %cmp10.not, i16 %1, i16 %2
  %inc = add nuw nsw i8 %i.036, 1
  %cmp = icmp ult i8 %inc, 16
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %crc.addr.0.lcssa = phi i16 [ %crc.addr.2, %for.body ]
  ret i16 %crc.addr.0.lcssa
}


; Two shifts per iteration. Check that the ValueBits are correctly mismatched
; CHECK-NOT: loop-idiom CRCRegonize: This looks like crc!
define signext i16 @crc16_doubleshift(i16 %crcValue, i8 %newByte) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.017 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %newByte.addr.016 = phi i8 [ %newByte, %entry ], [ %shl7, %for.body ]
  %crcValue.addr.015 = phi i16 [ %crcValue, %entry ], [ %crcValue.addr.1, %for.body ]
  %and = lshr i16 %crcValue.addr.015, 8
  %conv2 = zext i8 %newByte.addr.016 to i16
  %shr14 = xor i16 %conv2, %and
  %xor = and i16 %shr14, 128
  %tobool.not = icmp eq i16 %xor, 0
  %shlone = shl i16 %crcValue.addr.015, 1
  %shl = lshr i16 %shlone, 1
  %xor4 = xor i16 %shl, 258
  %crcValue.addr.1 = select i1 %tobool.not, i16 %shl, i16 %xor4
  %shl7 = shl i8 %newByte.addr.016, 1
  %inc = add nuw nsw i8 %i.017, 1
  %cmp = icmp ult i8 %inc, 8
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %crcValue.addr.0.lcssa = phi i16 [ %crcValue.addr.1, %for.body ]
  ret i16 %crcValue.addr.0.lcssa
}

; CHECK: loop-idiom CRCRegonize: ICmp RHS is not checking [M/L]SB
define signext i16 @crc16_not_check_sb(i16 %crcValue, i8 %newByte) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.017 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %newByte.addr.016 = phi i8 [ %newByte, %entry ], [ %shl7, %for.body ]
  %crcValue.addr.015 = phi i16 [ %crcValue, %entry ], [ %crcValue.addr.1, %for.body ]
  %and = lshr i16 %crcValue.addr.015, 8
  %conv2 = zext i8 %newByte.addr.016 to i16
  %shr14 = xor i16 %conv2, %and
  %xor = and i16 %shr14, 128
  %tobool.not = icmp eq i16 %xor, 2
  %shl = shl i16 %crcValue.addr.015, 1
  %xor4 = xor i16 %shl, 258
  %crcValue.addr.1 = select i1 %tobool.not, i16 %shl, i16 %xor4
  %shl7 = shl i8 %newByte.addr.016, 1
  %inc = add nuw nsw i8 %i.017, 1
  %cmp = icmp ult i8 %inc, 8
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %crcValue.addr.0.lcssa = phi i16 [ %crcValue.addr.1, %for.body ]
  ret i16 %crcValue.addr.0.lcssa
}
