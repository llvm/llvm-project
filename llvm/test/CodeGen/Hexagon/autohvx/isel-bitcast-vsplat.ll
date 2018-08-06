; RUN: llc -march=hexagon < %s | FileCheck %s

; This testcase exposed a problem with a previous handling of selecting
; constant vectors (for vdelta). Originally a bitcast of a vsplat was
; created (both being ISD, not machine nodes). Selection of vsplat relies
; on its return type, and there was no way to get these nodes to be
; selected in the right order, without getting the main selection algorithm
; confused.

; Make sure this compiles successfully.
; CHECK: call f1

target triple = "hexagon"

%s.0 = type { %s.1 }
%s.1 = type { i32, i8* }
%s.2 = type { i8, i8, [16 x i8], i8, [16 x i8] }

; Function Attrs: nounwind
define dso_local zeroext i8 @f0(i8 zeroext %a0, %s.2* nocapture readonly %a1, i8 signext %a2) local_unnamed_addr #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  %v0 = load <64 x i8>, <64 x i8>* undef, align 1
  %v1 = icmp ult <64 x i8> %v0, <i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52, i8 52>
  %v2 = xor <64 x i1> %v1, zeroinitializer
  %v3 = select <64 x i1> %v2, <64 x i32> undef, <64 x i32> zeroinitializer
  %v4 = select <64 x i1> zeroinitializer, <64 x i32> <i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000, i32 304000>, <64 x i32> %v3
  %v5 = add <64 x i32> %v4, zeroinitializer
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v6 = phi <64 x i32> [ undef, %b0 ], [ %v5, %b1 ]
  %v7 = add <64 x i32> %v6, undef
  %v8 = add <64 x i32> %v7, undef
  %v9 = add <64 x i32> %v8, undef
  %v10 = add <64 x i32> %v9, undef
  %v11 = add <64 x i32> %v10, undef
  %v12 = add <64 x i32> %v11, undef
  %v13 = extractelement <64 x i32> %v12, i32 0
  tail call void @f1(%s.0* null, i32 undef, i32 undef, i32 %v13, i32 undef) #2
  unreachable
}

declare dso_local void @f1(%s.0*, i32, i32, i32, i32) local_unnamed_addr #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
attributes #1 = { "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
attributes #2 = { nounwind }
