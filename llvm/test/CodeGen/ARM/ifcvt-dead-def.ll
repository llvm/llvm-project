; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-unknown-unknown"

%struct.ref_s = type { %union.v, i16, i16 }
%union.v = type { i32 }
%struct.gs_color_s = type { i16, i16, i16, i16, i8, i8 }

; In this case, the if converter was cloning the return instruction so that we had
;   r2 = ...
;   return [pred] r2<dead,def>
;   ldr <r2, kill>
;   return
; The problem here was that the dead def on the first return was making the machine verifier
; think that the load read from an undefined register.  We need to remove the dead flag from
; the return, and also add an implicit use of the prior value of r2.

; CHECK: ldrh

; Function Attrs: minsize nounwind optsize ssp
define i32 @test(%struct.ref_s* %pref1, %struct.ref_s* %pref2, %struct.gs_color_s** %tmp152) #0 {
bb:
  %nref = alloca %struct.ref_s, align 4
  %tmp46 = call %struct.ref_s* @name_string_ref(%struct.ref_s* %pref1, %struct.ref_s* %nref) #2
  %tmp153 = load %struct.gs_color_s*, %struct.gs_color_s** %tmp152, align 4
  %tmp154 = bitcast %struct.ref_s* %pref2 to %struct.gs_color_s**
  %tmp155 = load %struct.gs_color_s*, %struct.gs_color_s** %tmp154, align 4
  %tmp162 = getelementptr inbounds %struct.gs_color_s, %struct.gs_color_s* %tmp153, i32 0, i32 1
  %tmp163 = load i16, i16* %tmp162, align 2
  %tmp164 = getelementptr inbounds %struct.gs_color_s, %struct.gs_color_s* %tmp155, i32 0, i32 1
  %tmp165 = load i16, i16* %tmp164, align 2
  %tmp166 = icmp eq i16 %tmp163, %tmp165
  br i1 %tmp166, label %bb167, label %bb173

bb167:                                            ; preds = %bb
  %tmp168 = getelementptr inbounds %struct.gs_color_s, %struct.gs_color_s* %tmp153, i32 0, i32 2
  %tmp169 = load i16, i16* %tmp168, align 2
  %tmp170 = getelementptr inbounds %struct.gs_color_s, %struct.gs_color_s* %tmp155, i32 0, i32 2
  %tmp171 = load i16, i16* %tmp170, align 2
  %tmp172 = icmp eq i16 %tmp169, %tmp171
  br label %bb173

bb173:                                            ; preds = %bb167, %bb
  %tmp174 = phi i1 [ false, %bb ], [ %tmp172, %bb167 ]
  %tmp175 = zext i1 %tmp174 to i32
  ret i32 %tmp175
}

; Function Attrs: minsize optsize
declare %struct.ref_s* @name_string_ref(%struct.ref_s*, %struct.ref_s*) #1

attributes #0 = { minsize nounwind optsize }
attributes #1 = { minsize optsize }
attributes #2 = { minsize nounwind optsize }
