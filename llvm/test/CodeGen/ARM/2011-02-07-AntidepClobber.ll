; RUN: llc < %s -asm-verbose=false -O3  -mtriple=armv5e-none-linux-gnueabi | FileCheck %s
; PR8986: PostRA antidependence breaker must respect "earlyclobber".
; armv5e generates mulv5 that cannot used the same reg for src/dest.

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32"
target triple = "armv5e-none-linux-gnueabi"

define hidden fastcc void @storeAtts() nounwind {
entry:
  %.SV116 = alloca i8**
  br i1 undef, label %meshBB520, label %meshBB464

bb15:                                             ; preds = %meshBB424
  br i1 undef, label %bb216, label %meshBB396

bb22:                                             ; preds = %meshBB396
  br label %cBB564

cBB564:                                           ; preds = %cBB564, %bb22
  br label %cBB564

poolStoreString.exit.thread:                      ; preds = %meshBB424
  ret void

bb78:                                             ; preds = %meshBB412
  unreachable

bb129:                                            ; preds = %meshBB540
  br i1 undef, label %bb131.loopexit, label %meshBB540

bb131.loopexit:                                   ; preds = %bb129
  br label %bb131

bb131:                                            ; preds = %bb135, %bb131.loopexit
  br i1 undef, label %bb134, label %meshBB396

bb134:                                            ; preds = %bb131
  unreachable

bb135:                                            ; preds = %meshBB396
  %uriHash.1.phi.load = load i32* undef
  %.load120 = load i8*** %.SV116
  %.phi24 = load i8* null
  %.phi26 = load i8** null
  store i8 %.phi24, i8* %.phi26, align 1
  %0 = getelementptr inbounds i8* %.phi26, i32 1
  store i8* %0, i8** %.load120, align 4
  ; CHECK: mul [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{(lr|r[0-9]+)$}}
  %1 = mul i32 %uriHash.1.phi.load, 1000003
  %2 = xor i32 0, %1
  store i32 %2, i32* null
  %3 = load i8* null, align 1
  %4 = icmp eq i8 %3, 0
  store i8* %0, i8** undef
  br i1 %4, label %meshBB472, label %bb131

bb212:                                            ; preds = %meshBB540
  unreachable

bb216:                                            ; preds = %bb15
  ret void

meshBB396:                                        ; preds = %bb131, %bb15
  br i1 undef, label %bb135, label %bb22

meshBB412:                                        ; preds = %meshBB464
  br i1 undef, label %meshBB504, label %bb78

meshBB424:                                        ; preds = %meshBB464
  br i1 undef, label %poolStoreString.exit.thread, label %bb15

meshBB464:                                        ; preds = %entry
  br i1 undef, label %meshBB424, label %meshBB412

meshBB472:                                        ; preds = %meshBB504, %bb135
  unreachable

meshBB504:                                        ; preds = %meshBB412
  br label %meshBB472

meshBB520:                                        ; preds = %entry
  br label %meshBB540

meshBB540:                                        ; preds = %meshBB520, %bb129
  br i1 undef, label %bb212, label %bb129
}
