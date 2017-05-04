; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -join-liveintervals=false -verify-machineinstrs | FileCheck %s
;
; Check that copyPhysReg() properly adds impl-use operands of the super
; register while lowering a COPY of a GR128 bit reg.

define void @autogen_SD5585(i32*, i64) {
; CHECK: .text
BB:
  %L5 = load i1, i1* undef
  %I8 = insertelement <8 x i64> undef, i64 %1, i32 3
  %I21 = insertelement <8 x i64> zeroinitializer, i64 475435, i32 5
  br label %CF290

CF290:                                            ; preds = %CF290, %BB
  %B29 = urem <8 x i64> %I8, %I21
  %Cmp31 = icmp sge i1 undef, undef
  br i1 %Cmp31, label %CF290, label %CF296

CF296:                                            ; preds = %CF290
  %FC36 = sitofp <8 x i64> %B29 to <8 x double>
  br label %CF302

CF302:                                            ; preds = %CF307, %CF296
  %Shuff49 = shufflevector <8 x i64> undef, <8 x i64> zeroinitializer, <8 x i32> <i32 undef, i32 9, i32 11, i32 undef, i32 15, i32 1, i32 3, i32 5>
  %L69 = load i16, i16* undef
  br label %CF307

CF307:                                            ; preds = %CF302
  %Cmp84 = icmp ne i16 undef, %L69
  br i1 %Cmp84, label %CF302, label %CF301

CF301:                                            ; preds = %CF307
  %B126 = or i32 514315, undef
  br label %CF280

CF280:                                            ; preds = %CF280, %CF301
  %I139 = insertelement <8 x i64> %Shuff49, i64 undef, i32 2
  %B155 = udiv <8 x i64> %I8, %I139
  %Cmp157 = icmp ne i64 -1, undef
  br i1 %Cmp157, label %CF280, label %CF281

CF281:                                            ; preds = %CF280
  %Cmp164 = icmp slt i1 %L5, %Cmp84
  br label %CF282

CF282:                                            ; preds = %CF304, %CF281
  br label %CF289

CF289:                                            ; preds = %CF289, %CF282
  store i32 %B126, i32* %0
  %Cmp219 = icmp slt i64 undef, undef
  br i1 %Cmp219, label %CF289, label %CF304

CF304:                                            ; preds = %CF289
  %Cmp234 = icmp ult i64 0, undef
  br i1 %Cmp234, label %CF282, label %CF283

CF283:                                            ; preds = %CF308, %CF283, %CF304
  %E251 = extractelement <8 x i64> %B155, i32 0
  br i1 undef, label %CF283, label %CF308

CF308:                                            ; preds = %CF283
  store i1 %Cmp164, i1* undef
  br i1 undef, label %CF283, label %CF293

CF293:                                            ; preds = %CF308
  ret void
}
