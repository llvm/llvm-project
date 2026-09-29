; RUN: llc %s -o - -verify-machineinstrs -fast-isel=true -mattr=+vfp4 -mattr=+neon | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios8.0.0"

%union.DV = type { <2 x double> }

; Fast-ISel was incorrectly trying to codegen <2 x double> adds and returning only a single vadds
; Check that we generate the 2 vaddd's we expect

; CHECK: vadd.f64
; CHECK: vadd.f64

define i32 @main(i32 %argc, ptr nocapture readnone %Argv, <2 x double> %tmp31) {
bb:
  %Ad = alloca %union.DV, align 16
  %tmp33 = fadd <2 x double> %tmp31, %tmp31
  br label %bb37

bb37:                                             ; preds = %bb37, %bb
  %i.02 = phi i32 [ 0, %bb ], [ %tmp38, %bb37 ]
  store <2 x double> %tmp33, ptr %Ad, align 16
  %tmp38 = add nuw nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %tmp38, 500000
  br i1 %exitcond, label %bb39, label %bb37

bb39:                                             ; preds = %bb37
  call fastcc void @printDV(ptr %Ad)
  ret i32 0
}

declare hidden fastcc void @printDV(ptr nocapture readonly)
