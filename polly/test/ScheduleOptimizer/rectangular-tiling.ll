; RUN: opt %loadPolly -polly-tile-sizes=256,16                                                                                                                                        -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-tile-sizes=256,16 -polly-tiling=false                                                                                                                    -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=NOTILING
; RUN: opt %loadPolly -polly-tile-sizes=256,16 -polly-2nd-level-tiling -polly-2nd-level-tile-sizes=16,8                                                                               -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=TWOLEVEL
; RUN: opt %loadPolly -polly-tile-sizes=256,16 -polly-2nd-level-tiling -polly-2nd-level-tile-sizes=16,8 -polly-register-tiling                                                        -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=TWO-PLUS-REGISTER
; RUN: opt %loadPolly -polly-tile-sizes=256,16 -polly-2nd-level-tiling -polly-2nd-level-tile-sizes=16,8 -polly-register-tiling -polly-register-tile-sizes=2,4 -polly-vectorizer=polly -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=TWO-PLUS-REGISTER-PLUS-VECTORIZATION

; CHECK: // 1st level tiling - Tiles
; CHECK: for (int c0 = 0; c0 <= 3; c0 += 1)
; CHECK:   for (int c1 = 0; c1 <= 31; c1 += 1)
; CHECK:     // 1st level tiling - Points
; CHECK:     for (int c2 = 0; c2 <= 255; c2 += 1)
; CHECK:       for (int c3 = 0; c3 <= 15; c3 += 1)
; CHECK:         Stmt_for_body3(256 * c0 + c2, 16 * c1 + c3);

; NOTILING: for (int c0 = 0; c0 <= 1023; c0 += 1)
; NOTILING:   for (int c1 = 0; c1 <= 511; c1 += 1)
; NOTILING:     Stmt_for_body3(c0, c1);


; TWOLEVEL: // 1st level tiling - Tiles
; TWOLEVEL: for (int c0 = 0; c0 <= 3; c0 += 1)
; TWOLEVEL:   for (int c1 = 0; c1 <= 31; c1 += 1)
; TWOLEVEL:     // 1st level tiling - Points
; TWOLEVEL:     // 2nd level tiling - Tiles
; TWOLEVEL:     for (int c2 = 0; c2 <= 15; c2 += 1)
; TWOLEVEL:       for (int c3 = 0; c3 <= 1; c3 += 1)
; TWOLEVEL:         // 2nd level tiling - Points
; TWOLEVEL:         for (int c4 = 0; c4 <= 15; c4 += 1)
; TWOLEVEL:           for (int c5 = 0; c5 <= 7; c5 += 1)
; TWOLEVEL:             Stmt_for_body3(256 * c0 + 16 * c2 + c4, 16 * c1 + 8 * c3 + c5);


; TWO-PLUS-REGISTER: // 1st level tiling - Tiles
; TWO-PLUS-REGISTER: for (int c0 = 0; c0 <= 3; c0 += 1)
; TWO-PLUS-REGISTER:   for (int c1 = 0; c1 <= 31; c1 += 1)
; TWO-PLUS-REGISTER:     // 1st level tiling - Points
; TWO-PLUS-REGISTER:     // 2nd level tiling - Tiles
; TWO-PLUS-REGISTER:     for (int c2 = 0; c2 <= 15; c2 += 1)
; TWO-PLUS-REGISTER:       for (int c3 = 0; c3 <= 1; c3 += 1)
; TWO-PLUS-REGISTER:         // 2nd level tiling - Points
; TWO-PLUS-REGISTER:         // Register tiling - Tiles
; TWO-PLUS-REGISTER:         for (int c4 = 0; c4 <= 7; c4 += 1)
; TWO-PLUS-REGISTER:           for (int c5 = 0; c5 <= 3; c5 += 1)
; TWO-PLUS-REGISTER:             // Register tiling - Points
; TWO-PLUS-REGISTER:             {
; TWO-PLUS-REGISTER:               Stmt_for_body3(256 * c0 + 16 * c2 + 2 * c4, 16 * c1 + 8 * c3 + 2 * c5);
; TWO-PLUS-REGISTER:               Stmt_for_body3(256 * c0 + 16 * c2 + 2 * c4, 16 * c1 + 8 * c3 + 2 * c5 + 1);
; TWO-PLUS-REGISTER:               Stmt_for_body3(256 * c0 + 16 * c2 + 2 * c4 + 1, 16 * c1 + 8 * c3 + 2 * c5);
; TWO-PLUS-REGISTER:               Stmt_for_body3(256 * c0 + 16 * c2 + 2 * c4 + 1, 16 * c1 + 8 * c3 + 2 * c5 + 1);
; TWO-PLUS-REGISTER:             }

; TWO-PLUS-REGISTER-PLUS-VECTORIZATION: #pragma known-parallel
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION: for (int c0 = 0; c0 <= 3; c0 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:   for (int c1 = 0; c1 <= 31; c1 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:     for (int c2 = 0; c2 <= 15; c2 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:       for (int c3 = 0; c3 <= 1; c3 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:         for (int c4 = 0; c4 <= 7; c4 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:           for (int c5 = 0; c5 <= 1; c5 += 1) {
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:             // SIMD
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:             for (int c8 = 0; c8 <= 3; c8 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:               Stmt_for_body3(256 * c0 + 16 * c2 + 2 * c4, 16 * c1 + 8 * c3 + 4 * c5 + c8);
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:             // SIMD
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:             for (int c8 = 0; c8 <= 3; c8 += 1)
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:               Stmt_for_body3(256 * c0 + 16 * c2 + 2 * c4 + 1, 16 * c1 + 8 * c3 + 4 * c5 + c8);
; TWO-PLUS-REGISTER-PLUS-VECTORIZATION:           }

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

; Function Attrs: nounwind
define void @rect([512 x i32]* %A) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.inc5, %entry.split
  %i.0 = phi i32 [ 0, %entry.split ], [ %inc6, %for.inc5 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.body3
  %j.0 = phi i32 [ 0, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %mul = mul nsw i32 %j.0, %i.0
  %rem = srem i32 %mul, 42
  %arrayidx4 = getelementptr inbounds [512 x i32], [512 x i32]* %A, i32 %i.0, i32 %j.0
  store i32 %rem, i32* %arrayidx4, align 4
  %inc = add nsw i32 %j.0, 1
  %cmp2 = icmp slt i32 %inc, 512
  br i1 %cmp2, label %for.body3, label %for.inc5

for.inc5:                                         ; preds = %for.body3
  %inc6 = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc6, 1024
  br i1 %cmp, label %for.body3.lr.ph, label %for.end7

for.end7:                                         ; preds = %for.inc5
  ret void
}
