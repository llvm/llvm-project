; RUN: llc < %s
; PR6363
;
; This test case creates a phi join register with a single definition. The other
; predecessor blocks are implicit-def.
;
; If LiveIntervalAnalysis fails to recognize this as a phi join, the coalescer
; will detect an infinity valno loop.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @decode(ptr nocapture %input, i32 %offset, ptr nocapture %output) nounwind {
entry:
  br i1 undef, label %meshBB86, label %meshBB102

bb:                                               ; preds = %meshBB106, %meshBB102
  br i1 false, label %bb9, label %meshBB90

bb.nph:                                           ; preds = %meshBB90
  br label %meshBB114

bb.nph.fragment:                                  ; preds = %meshBB114
  br label %meshBB118

bb1.fragment:                                     ; preds = %meshBB118
  br i1 false, label %bb2, label %bb3

bb2:                                              ; preds = %bb1.fragment
  br label %meshBB74

bb2.fragment15:                                   ; preds = %meshBB74
  br label %meshBB98

bb3:                                              ; preds = %bb1.fragment
  br i1 undef, label %meshBB, label %meshBB102

bb4:                                              ; preds = %meshBB
  br label %meshBB118

bb4.fragment:                                     ; preds = %meshBB118
  br label %meshBB82

bb5:                                              ; preds = %meshBB102, %meshBB82
  br i1 false, label %bb6, label %bb7

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %meshBB98, %bb6, %bb5
  br label %meshBB114

bb7.fragment:                                     ; preds = %meshBB114
  br i1 undef, label %meshBB74, label %bb9

bb9:                                              ; preds = %bb7.fragment, %bb
  br label %bb1.i23

bb1.i23:                                          ; preds = %meshBB110, %bb9
  br i1 undef, label %meshBB106, label %meshBB110

skip_to_newline.exit26:                           ; preds = %meshBB106
  br label %meshBB86

skip_to_newline.exit26.fragment:                  ; preds = %meshBB86
  br i1 false, label %meshBB90, label %meshBB106

bb11.fragment:                                    ; preds = %meshBB90, %meshBB86
  br label %meshBB122

bb1.i:                                            ; preds = %meshBB122, %meshBB
  %ooffset.2.lcssa.phi.SV.phi203 = phi i32 [ 0, %meshBB122 ], [ %ooffset.2.lcssa.phi.SV.phi233, %meshBB ] ; <i32> [#uses=1]
  br label %meshBB98

bb1.i.fragment:                                   ; preds = %meshBB98
  br i1 undef, label %meshBB78, label %meshBB

skip_to_newline.exit:                             ; preds = %meshBB78
  br i1 undef, label %bb12, label %meshBB110

bb12:                                             ; preds = %skip_to_newline.exit
  br label %meshBB94

bb12.fragment:                                    ; preds = %meshBB94
  br i1 false, label %bb13, label %meshBB78

bb13:                                             ; preds = %bb12.fragment
  br label %meshBB82

bb13.fragment:                                    ; preds = %meshBB82
  br i1 undef, label %meshBB94, label %meshBB122

bb14:                                             ; preds = %meshBB94
  ret i32 %ooffset.2.lcssa.phi.SV.phi250

bb15:                                             ; preds = %meshBB122, %meshBB110, %meshBB78
  unreachable

meshBB:                                           ; preds = %bb1.i.fragment, %bb3
  %ooffset.2.lcssa.phi.SV.phi233 = phi i32 [ undef, %bb3 ], [ %ooffset.2.lcssa.phi.SV.phi209, %bb1.i.fragment ] ; <i32> [#uses=1]
  br i1 undef, label %bb1.i, label %bb4

meshBB74:                                         ; preds = %bb7.fragment, %bb2
  br i1 false, label %meshBB118, label %bb2.fragment15

meshBB78:                                         ; preds = %bb12.fragment, %bb1.i.fragment
  %ooffset.2.lcssa.phi.SV.phi239 = phi i32 [ %ooffset.2.lcssa.phi.SV.phi209, %bb1.i.fragment ], [ %ooffset.2.lcssa.phi.SV.phi250, %bb12.fragment ] ; <i32> [#uses=1]
  br i1 false, label %bb15, label %skip_to_newline.exit

meshBB82:                                         ; preds = %bb13, %bb4.fragment
  br i1 false, label %bb5, label %bb13.fragment

meshBB86:                                         ; preds = %skip_to_newline.exit26, %entry
  br i1 undef, label %skip_to_newline.exit26.fragment, label %bb11.fragment

meshBB90:                                         ; preds = %skip_to_newline.exit26.fragment, %bb
  br i1 false, label %bb11.fragment, label %bb.nph

meshBB94:                                         ; preds = %bb13.fragment, %bb12
  %ooffset.2.lcssa.phi.SV.phi250 = phi i32 [ 0, %bb13.fragment ], [ %ooffset.2.lcssa.phi.SV.phi239, %bb12 ] ; <i32> [#uses=2]
  br i1 false, label %bb12.fragment, label %bb14

meshBB98:                                         ; preds = %bb1.i, %bb2.fragment15
  %ooffset.2.lcssa.phi.SV.phi209 = phi i32 [ undef, %bb2.fragment15 ], [ %ooffset.2.lcssa.phi.SV.phi203, %bb1.i ] ; <i32> [#uses=2]
  br i1 undef, label %bb1.i.fragment, label %bb7

meshBB102:                                        ; preds = %bb3, %entry
  br i1 undef, label %bb5, label %bb

meshBB106:                                        ; preds = %skip_to_newline.exit26.fragment, %bb1.i23
  br i1 undef, label %bb, label %skip_to_newline.exit26

meshBB110:                                        ; preds = %skip_to_newline.exit, %bb1.i23
  br i1 false, label %bb15, label %bb1.i23

meshBB114:                                        ; preds = %bb7, %bb.nph
  %meshStackVariable115.phi = phi i32 [ 19, %bb7 ], [ 8, %bb.nph ] ; <i32> [#uses=0]
  br i1 undef, label %bb.nph.fragment, label %bb7.fragment

meshBB118:                                        ; preds = %meshBB74, %bb4, %bb.nph.fragment
  %meshCmp121 = icmp eq i32 undef, 10             ; <i1> [#uses=1]
  br i1 %meshCmp121, label %bb4.fragment, label %bb1.fragment

meshBB122:                                        ; preds = %bb13.fragment, %bb11.fragment
  br i1 false, label %bb1.i, label %bb15
}
