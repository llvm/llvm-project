; RUN: opt -passes=loop-idiom < %s -S -debug -recognize-crc 2>&1 | FileCheck %s

; CRC 8 bit, data 8 bit
; CHECK: GeneratorPolynomial: 29
; CHECK: CRC Size: 8
; CHECK: Reversed: 0
; CHECK: loop-idiom CRCRecognize: This looks like crc!
define dso_local zeroext i8 @crc8_loop(ptr noundef %data, i32 noundef %length) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup7, %entry
  %crc.0 = phi i8 [ 0, %entry ], [ %crc.1.lcssa, %for.cond.cleanup7 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc20, %for.cond.cleanup7 ]
  %cmp = icmp ult i32 %i.0, %length
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %crc.0.lcssa = phi i8 [ %crc.0, %for.cond ]
  ret i8 %crc.0.lcssa

for.body:                                         ; preds = %for.cond
  %add.ptr = getelementptr inbounds i8, ptr %data, i32 %i.0
  %0 = load i8, ptr %add.ptr, align 1
  %xor29 = xor i8 %0, %crc.0
  br label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8
  %crc.1.lcssa = phi i8 [ %crc.2, %for.body8 ]
  %inc20 = add i32 %i.0, 1
  br label %for.cond

for.body8:                                        ; preds = %for.body, %for.body8
  %i3.032 = phi i32 [ 0, %for.body ], [ %inc, %for.body8 ]
  %crc.131 = phi i8 [ %xor29, %for.body ], [ %crc.2, %for.body8 ]
  %shl = shl i8 %crc.131, 1
  %xor14 = xor i8 %shl, 29
  %cmp10.not30 = icmp slt i8 %crc.131, 0
  %crc.2 = select i1 %cmp10.not30, i8 %xor14, i8 %shl
  %inc = add nuw nsw i32 %i3.032, 1
  %cmp5 = icmp ult i32 %inc, 8
  br i1 %cmp5, label %for.body8, label %for.cond.cleanup7
}

; CRC16, 8 bit data
; CHECK: Input CRC: i16 %crc
; CHECK: Output CRC:   %crc.addr.2
; CHECK: GeneratorPolynomial: 32773
; CHECK: CRC Size: 16
; CHECK: Reversed: 1
; CHECK: Data Input: i8 %data
; CHECK: Data Size: 8
define i16 @crc16_reversed(i8 %data, i16 %crc) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.036 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %crc.addr.035 = phi i16 [ %crc, %entry ], [ %crc.addr.2, %for.body ]
  %data.addr.034 = phi i8 [ %data, %entry ], [ %1, %for.body ]
  %0 = trunc i16 %crc.addr.035 to i8
  %and33 = xor i8 %0, %data.addr.034
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

; CRC16 xor outside loop
; CHECK: loop-idiom CRCRecognize: This looks like crc!
define dso_local zeroext i16 @crc16_xor_outside(i16 %crc, i8 %data) {
entry:
  %conv2 = zext i8 %data to i16
  %shl = shl nuw i16 %conv2, 8
  %xor = xor i16 %shl, %crc
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.020 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %crc.addr.019 = phi i16 [ %xor, %entry ], [ %crc.addr.1, %for.body ]
  %shl7 = shl i16 %crc.addr.019, 1
  %xor8 = xor i16 %shl7, 4129
  %tobool.not18 = icmp slt i16 %crc.addr.019, 0
  %crc.addr.1 = select i1 %tobool.not18, i16 %xor8, i16 %shl7
  %inc = add nuw nsw i32 %i.020, 1
  %cmp = icmp ult i32 %inc, 8
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %crc.addr.0.lcssa = phi i16 [ %crc.addr.1, %for.body ]
  ret i16 %crc.addr.0.lcssa
}

; CRC size 32 xor inside in a byte loop
; CHECK: GeneratorPolynomial: 270598144
; CHECK: CRC Size: 32
; CHECK: loop-idiom CRCRecognize: This looks like crc!
define i16 @crc32_reversed(ptr %data_p, i16 %length) {
entry:
  %cmp = icmp eq i16 %length, 0
  br i1 %cmp, label %cleanup, label %do.body.preheader

do.body.preheader:                                ; preds = %entry
  br label %do.body

do.body:                                          ; preds = %do.body.preheader, %do.cond
  %data_p.addr.0 = phi ptr [ %incdec.ptr, %do.cond ], [ %data_p, %do.body.preheader ]
  %length.addr.0 = phi i16 [ %dec, %do.cond ], [ %length, %do.body.preheader ]
  %crc.0 = phi i32 [ %crc.1.lcssa, %do.cond ], [ 65535, %do.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, ptr %data_p.addr.0, i64 1
  %0 = load i8, ptr %data_p.addr.0, align 1
  %conv3 = zext i8 %0 to i32
  br label %for.body

for.body:                                         ; preds = %do.body, %for.body
  %crc.135 = phi i32 [ %crc.0, %do.body ], [ %crc.2, %for.body ]
  %data.034 = phi i32 [ %conv3, %do.body ], [ %shr13, %for.body ]
  %i.033 = phi i8 [ 0, %do.body ], [ %inc, %for.body ]
  %and732 = xor i32 %crc.135, %data.034
  %xor = and i32 %and732, 1
  %tobool.not = icmp eq i32 %xor, 0
  %shr = lshr i32 %crc.135, 1
  %xor10 = xor i32 %shr, 33800
  %crc.2 = select i1 %tobool.not, i32 %shr, i32 %xor10
  %inc = add nuw nsw i8 %i.033, 1
  %shr13 = lshr i32 %data.034, 1
  %cmp5 = icmp ult i8 %inc, 8
  br i1 %cmp5, label %for.body, label %do.cond

do.cond:                                          ; preds = %for.body
  %crc.1.lcssa = phi i32 [ %crc.2, %for.body ]
  %dec = add i16 %length.addr.0, -1
  %tobool14.not = icmp eq i16 %dec, 0
  br i1 %tobool14.not, label %do.end, label %do.body

do.end:                                           ; preds = %do.cond
  %crc.1.lcssa.lcssa = phi i32 [ %crc.1.lcssa, %do.cond ]
  %not15 = xor i32 %crc.1.lcssa.lcssa, -1
  %shl = shl i32 %not15, 8
  %shr16 = lshr i32 %not15, 8
  %and17 = and i32 %shr16, 255
  %or = add nuw nsw i32 %and17, %shl
  %conv18 = trunc i32 %or to i16
  br label %cleanup

cleanup:                                          ; preds = %entry, %do.end
  %retval.0 = phi i16 [ %conv18, %do.end ], [ 0, %entry ]
  ret i16 %retval.0
}

; CRC16 
; CHECK: GeneratorPolynomial: 258
; CHECK: CRC Size: 16
; CHECK: Reversed: 0
; CHECK: Data Size: 8
; CHECK: loop-idiom CRCRecognize: This looks like crc!
define signext i16 @crc16(i16 %crcValue, i8 %newByte) {
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

; CHECK: @crctable.i16.32773.reversed = private constant [256 x i16] [i16 0, i16 -16191, i16 -15999, i16 320
; CHECK: @crctable.i16.4129 = private constant [256 x i16] [i16 0, i16 4129, i16 8258, i16 12387, i16 16516
; CHECK: @crctable.i32.270598144.reversed = private constant [256 x i32] [i32 0, i32 4489, i32 8978, i32 12955
; CHECK: @crctable.i16.258 = private constant [256 x i16] [i16 0, i16 258, i16 516, i16 774, i16 1032
