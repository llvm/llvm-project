; RUN: llc < %s | FileCheck %s
; Check that an overly large immediate created by SROA doesn't crash the
; legalizer.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct._GtkSheetRow = type { i32*, i32, i32, i32, %struct._GtkSheetButton, i32, i32 }
%struct._GtkSheetButton = type { i32, i32*, i32, i32*, i32 }

@a = common global %struct._GtkSheetRow* null, align 8

define void @fn1() nounwind uwtable ssp {
entry:
  %0 = load %struct._GtkSheetRow*, %struct._GtkSheetRow** @a, align 8
  %1 = bitcast %struct._GtkSheetRow* %0 to i576*
  %srcval2 = load i576, i576* %1, align 8
  %tobool = icmp ugt i576 %srcval2, 57586096570152913699974892898380567793532123114264532903689671329431521032595044740083720782129802971518987656109067457577065805510327036019308994315074097345724415
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i576 %srcval2, i576* %1, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void

; CHECK-LABEL: fn1:
; CHECK: jb
}
