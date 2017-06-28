; RUN: opt -S -codegenprepare -mtriple=i686-unknown-unknown   -data-layout=e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128 < %s | FileCheck %s --check-prefix=ALL --check-prefix=X32
; RUN: opt -S -codegenprepare -mtriple=x86_64-unknown-unknown -data-layout=e-m:o-i64:64-f80:128-n8:16:32:64-S128         < %s | FileCheck %s --check-prefix=ALL --check-prefix=X64

declare i32 @memcmp(i8* nocapture, i8* nocapture, i64)

define i32 @cmp2(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp2(
; ALL-NEXT:    [[TMP1:%.*]] = bitcast i8* %x to i16*
; ALL-NEXT:    [[TMP2:%.*]] = bitcast i8* %y to i16*
; ALL-NEXT:    [[TMP3:%.*]] = load i16, i16* [[TMP1]]
; ALL-NEXT:    [[TMP4:%.*]] = load i16, i16* [[TMP2]]
; ALL-NEXT:    [[TMP5:%.*]] = call i16 @llvm.bswap.i16(i16 [[TMP3]])
; ALL-NEXT:    [[TMP6:%.*]] = call i16 @llvm.bswap.i16(i16 [[TMP4]])
; ALL-NEXT:    [[TMP7:%.*]] = icmp ne i16 [[TMP5]], [[TMP6]]
; ALL-NEXT:    [[TMP8:%.*]] = icmp ult i16 [[TMP5]], [[TMP6]]
; ALL-NEXT:    [[TMP9:%.*]] = select i1 [[TMP8]], i32 -1, i32 1
; ALL-NEXT:    [[TMP10:%.*]] = select i1 [[TMP7]], i32 [[TMP9]], i32 0
; ALL-NEXT:    ret i32 [[TMP10]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 2)
  ret i32 %call
}

define i32 @cmp3(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp3(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 3)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 3)
  ret i32 %call
}

define i32 @cmp4(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp4(
; ALL-NEXT:    [[TMP1:%.*]] = bitcast i8* %x to i32*
; ALL-NEXT:    [[TMP2:%.*]] = bitcast i8* %y to i32*
; ALL-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP1]]
; ALL-NEXT:    [[TMP4:%.*]] = load i32, i32* [[TMP2]]
; ALL-NEXT:    [[TMP5:%.*]] = call i32 @llvm.bswap.i32(i32 [[TMP3]])
; ALL-NEXT:    [[TMP6:%.*]] = call i32 @llvm.bswap.i32(i32 [[TMP4]])
; ALL-NEXT:    [[TMP7:%.*]] = icmp ne i32 [[TMP5]], [[TMP6]]
; ALL-NEXT:    [[TMP8:%.*]] = icmp ult i32 [[TMP5]], [[TMP6]]
; ALL-NEXT:    [[TMP9:%.*]] = select i1 [[TMP8]], i32 -1, i32 1
; ALL-NEXT:    [[TMP10:%.*]] = select i1 [[TMP7]], i32 [[TMP9]], i32 0
; ALL-NEXT:    ret i32 [[TMP10]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 4)
  ret i32 %call
}

define i32 @cmp5(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp5(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 5)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 5)
  ret i32 %call
}

define i32 @cmp6(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp6(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 6)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 6)
  ret i32 %call
}

define i32 @cmp7(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp7(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 7)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 7)
  ret i32 %call
}

define i32 @cmp8(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; X32-LABEL: @cmp8(
; X32-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 8)
; X32-NEXT:    ret i32 [[CALL]]
;
; X64-LABEL: @cmp8(
; X64-NEXT:    [[TMP1:%.*]] = bitcast i8* %x to i64*
; X64-NEXT:    [[TMP2:%.*]] = bitcast i8* %y to i64*
; X64-NEXT:    [[TMP3:%.*]] = load i64, i64* [[TMP1]]
; X64-NEXT:    [[TMP4:%.*]] = load i64, i64* [[TMP2]]
; X64-NEXT:    [[TMP5:%.*]] = call i64 @llvm.bswap.i64(i64 [[TMP3]])
; X64-NEXT:    [[TMP6:%.*]] = call i64 @llvm.bswap.i64(i64 [[TMP4]])
; X64-NEXT:    [[TMP7:%.*]] = icmp ne i64 [[TMP5]], [[TMP6]]
; X64-NEXT:    [[TMP8:%.*]] = icmp ult i64 [[TMP5]], [[TMP6]]
; X64-NEXT:    [[TMP9:%.*]] = select i1 [[TMP8]], i32 -1, i32 1
; X64-NEXT:    [[TMP10:%.*]] = select i1 [[TMP7]], i32 [[TMP9]], i32 0
; X64-NEXT:    ret i32 [[TMP10]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 8)
  ret i32 %call
}

define i32 @cmp9(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp9(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 9)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 9)
  ret i32 %call
}

define i32 @cmp10(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp10(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 10)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 10)
  ret i32 %call
}

define i32 @cmp11(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp11(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 11)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 11)
  ret i32 %call
}

define i32 @cmp12(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp12(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 12)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 12)
  ret i32 %call
}

define i32 @cmp13(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp13(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 13)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 13)
  ret i32 %call
}

define i32 @cmp14(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp14(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 14)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 14)
  ret i32 %call
}

define i32 @cmp15(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp15(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 15)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 15)
  ret i32 %call
}

define i32 @cmp16(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp16(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 16)
; ALL-NEXT:    ret i32 [[CALL]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 16)
  ret i32 %call
}

define i32 @cmp_eq2(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq2(
; ALL-NEXT:    [[TMP1:%.*]] = bitcast i8* %x to i16*
; ALL-NEXT:    [[TMP2:%.*]] = bitcast i8* %y to i16*
; ALL-NEXT:    [[TMP3:%.*]] = load i16, i16* [[TMP1]]
; ALL-NEXT:    [[TMP4:%.*]] = load i16, i16* [[TMP2]]
; ALL-NEXT:    [[TMP5:%.*]] = icmp ne i16 [[TMP3]], [[TMP4]]
; ALL-NEXT:    [[TMP6:%.*]] = zext i1 [[TMP5]] to i32
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP6]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 2)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq3(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq3(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 3)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 3)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq4(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq4(
; ALL-NEXT:    [[TMP1:%.*]] = bitcast i8* %x to i32*
; ALL-NEXT:    [[TMP2:%.*]] = bitcast i8* %y to i32*
; ALL-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP1]]
; ALL-NEXT:    [[TMP4:%.*]] = load i32, i32* [[TMP2]]
; ALL-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP3]], [[TMP4]]
; ALL-NEXT:    [[TMP6:%.*]] = zext i1 [[TMP5]] to i32
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP6]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 4)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq5(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq5(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 5)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 5)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq6(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq6(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 6)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 6)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq7(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq7(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 7)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 7)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq8(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; X32-LABEL: @cmp_eq8(
; X32-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 8)
; X32-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; X32-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; X32-NEXT:    ret i32 [[CONV]]
;
; X64-LABEL: @cmp_eq8(
; X64-NEXT:    [[TMP1:%.*]] = bitcast i8* %x to i64*
; X64-NEXT:    [[TMP2:%.*]] = bitcast i8* %y to i64*
; X64-NEXT:    [[TMP3:%.*]] = load i64, i64* [[TMP1]]
; X64-NEXT:    [[TMP4:%.*]] = load i64, i64* [[TMP2]]
; X64-NEXT:    [[TMP5:%.*]] = icmp ne i64 [[TMP3]], [[TMP4]]
; X64-NEXT:    [[TMP6:%.*]] = zext i1 [[TMP5]] to i32
; X64-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP6]], 0
; X64-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; X64-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 8)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq9(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq9(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 9)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 9)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq10(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq10(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 10)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 10)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq11(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq11(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 11)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 11)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq12(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq12(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 12)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 12)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq13(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq13(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 13)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 13)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq14(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq14(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 14)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 14)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq15(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq15(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 15)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 15)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @cmp_eq16(i8* nocapture readonly %x, i8* nocapture readonly %y)  {
; ALL-LABEL: @cmp_eq16(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* %y, i64 16)
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
; ALL-NEXT:    ret i32 [[CONV]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* %y, i64 16)
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

