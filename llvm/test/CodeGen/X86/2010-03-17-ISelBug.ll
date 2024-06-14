; RUN: llc < %s -mtriple=i386-apple-darwin5

; rdar://7761790

%"struct..0$_485" = type { i16, i16, i32 }
%union.PPToken = type { %"struct..0$_485" }
%struct.PPOperation = type { %union.PPToken, %union.PPToken, [6 x %union.PPToken], i32, i32, i32, [1 x i32], [0 x i8] }

define ptr @t() align 2 nounwind {
entry:
  %operation = alloca %struct.PPOperation, align 8 ; <ptr> [#uses=2]
  %0 = load ptr, ptr null, align 4  ; [#uses=1]
  %1 = ptrtoint ptr %0 to i32   ; <i32> [#uses=1]
  %2 = sub nsw i32 %1, undef                      ; <i32> [#uses=2]
  br i1 false, label %bb20, label %bb.nph380

bb20:                                             ; preds = %entry
  ret ptr null

bb.nph380:                                        ; preds = %entry
  %scevgep403 = getelementptr %struct.PPOperation, ptr %operation, i32 0, i32 1, i32 0, i32 2 ; <ptr> [#uses=1]
  %3 = ashr i32 %2, 1                             ; <i32> [#uses=1]
  %tmp405 = and i32 %3, -2                        ; <i32> [#uses=1]
  %scevgep408 = getelementptr %struct.PPOperation, ptr %operation, i32 0, i32 1, i32 0, i32 1 ; <ptr> [#uses=1]
  %tmp410 = and i32 %2, -4                        ; <i32> [#uses=1]
  br label %bb169

bb169:                                            ; preds = %bb169, %bb.nph380
  %index.6379 = phi i32 [ 0, %bb.nph380 ], [ %4, %bb169 ] ; <i32> [#uses=3]
  %tmp404 = mul i32 %index.6379, -2               ; <i32> [#uses=1]
  %tmp406 = add i32 %tmp405, %tmp404              ; <i32> [#uses=1]
  %scevgep407 = getelementptr i32, ptr %scevgep403, i32 %tmp406 ; <ptr> [#uses=1]
  %tmp409 = mul i32 %index.6379, -4               ; <i32> [#uses=1]
  %tmp411 = add i32 %tmp410, %tmp409              ; <i32> [#uses=1]
  %scevgep412 = getelementptr i16, ptr %scevgep408, i32 %tmp411 ; <ptr> [#uses=1]
  store i16 undef, ptr %scevgep412, align 2
  store i32 undef, ptr %scevgep407, align 4
  %4 = add nsw i32 %index.6379, 1                 ; <i32> [#uses=1]
  br label %bb169
}

; PR7368

%struct.bufBit_s = type { ptr, i8 }

define fastcc void @printSwipe(ptr nocapture %colourLines) nounwind {
entry:
  br label %for.body190
  
for.body261.i:                                    ; preds = %for.body261.i, %for.body190
  %line.3300.i = phi i32 [ undef, %for.body190 ], [ %add292.i, %for.body261.i ] ; <i32> [#uses=3]
  %conv268.i = and i32 %line.3300.i, 255          ; <i32> [#uses=1]
  %tmp278.i = getelementptr [2 x [256 x %struct.bufBit_s]], ptr %colourLines, i32 undef, i32 %pen.1100, i32 %conv268.i, i32 0 ; <ptr> [#uses=1]
  store ptr undef, ptr %tmp278.i
  %tmp338 = shl i32 %line.3300.i, 3               ; <i32> [#uses=1]
  %tmp339 = and i32 %tmp338, 2040                 ; <i32> [#uses=1]
  %tmp285.i = getelementptr i8, ptr %scevgep328, i32 %tmp339 ; <ptr> [#uses=1]
  store i8 undef, ptr %tmp285.i
  %add292.i = add nsw i32 0, %line.3300.i         ; <i32> [#uses=1]
  br i1 undef, label %for.body190, label %for.body261.i

for.body190:                                      ; preds = %for.body261.i, %for.body190, %bb.nph104
  %pen.1100 = phi i32 [ 0, %entry ], [ %inc230, %for.body261.i ], [ %inc230, %for.body190 ] ; <i32> [#uses=3]
  %scevgep328 = getelementptr [2 x [256 x %struct.bufBit_s]], ptr %colourLines, i32 undef, i32 %pen.1100, i32 0, i32 1 ; <ptr> [#uses=1]
  %inc230 = add i32 %pen.1100, 1                  ; <i32> [#uses=2]
  br i1 undef, label %for.body190, label %for.body261.i
}
