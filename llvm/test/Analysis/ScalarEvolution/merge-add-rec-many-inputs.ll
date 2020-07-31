; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; Check that isImpliedViaMerge wouldn't crash when trying to prove
; SCEVUnknown and AddRec with phi having many inputs
; CHECK: @foo

define void @foo(i1 %cond) {
osr.type.merge79:
  br label %bci_329

bci_329:                                          ; preds = %bci_337, %bci_326, %osr.type.merge79
  %local_7_ = phi i32 [ 0, %osr.type.merge79 ], [ %local_7_113, %bci_326 ], [ %local_7_, %bci_337 ]
  br i1 %cond, label %bci_360, label %bci_337

bci_360:                                          ; preds = %bci_329
  %0 = phi i32 [ %local_7_, %bci_329 ]
  %1 = icmp sge i32 %0, 451
  br i1 %1, label %bci_371, label %bci_326

bci_371:                                          ; preds = %bci_304, %bci_360
  %local_2_132 = phi i32 [ 0, %bci_360 ], [ %local_2_188, %bci_304 ]
  %local_7_137 = phi i32 [ 0, %bci_360 ], [ %local_7_217, %bci_304 ]
  %local_8_138 = phi i32 [ 0, %bci_360 ], [ %local_8_194, %bci_304 ]
  %local_11_141 = phi i32 [ 0, %bci_360 ], [ %local_11_197, %bci_304 ]
  %2 = phi i1 [ %1, %bci_360 ], [ false, %bci_304 ]
  br i1 %2, label %bci_562, label %never_deopt

bci_562:                                          ; preds = %done437, %done405, %done395, %bci_407, %not_zero322, %bci_221, %bci_371
  %local_2_164 = phi i32 [ %local_2_132, %bci_371 ], [ %9, %not_zero322 ], [ %local_2_188, %bci_407 ], [ %result396, %done395 ], [ %local_2_188, %bci_221 ], [ %local_2_188, %done405 ], [ %local_2_188, %done437 ]
  %local_7_169 = phi i32 [ %local_7_137, %bci_371 ], [ %local_7_217, %not_zero322 ], [ %local_7_217, %bci_407 ], [ %local_7_217, %done395 ], [ %local_7_217, %bci_221 ], [ %local_7_217, %done405 ], [ %local_7_217, %done437 ]
  %local_8_170 = phi i32 [ %local_8_138, %bci_371 ], [ %local_8_194, %not_zero322 ], [ %local_8_194, %bci_407 ], [ %local_8_194, %done395 ], [ %local_8_194, %bci_221 ], [ %local_8_194, %done405 ], [ %local_8_194, %done437 ]
  %local_11_173 = phi i32 [ %local_11_141, %bci_371 ], [ %local_11_197, %not_zero322 ], [ %local_11_197, %bci_407 ], [ %local_11_197, %done395 ], [ %local_11_197, %bci_221 ], [ %local_11_197, %done405 ], [ %local_11_197, %done437 ]
  br label %bci_604

bci_604:                                          ; preds = %not_subtype, %bci_565, %bci_562
  %local_2_188 = phi i32 [ %local_2_164, %bci_562 ], [ %9, %bci_565 ], [ %9, %not_subtype ]
  %local_7_193 = phi i32 [ %local_7_169, %bci_562 ], [ %local_7_217, %bci_565 ], [ %local_7_217, %not_subtype ]
  %local_8_194 = phi i32 [ %local_8_170, %bci_562 ], [ %local_8_194, %bci_565 ], [ %local_8_194, %not_subtype ]
  %local_11_197 = phi i32 [ %local_11_173, %bci_562 ], [ %local_11_197, %bci_565 ], [ %local_11_197, %not_subtype ]
  %3 = add i32 1, %local_7_193
  br label %bci_199

bci_199:                                          ; preds = %bci_591, %bci_604
  %local_7_217 = phi i32 [ %3, %bci_604 ], [ %6, %bci_591 ]
  %4 = mul i32 %local_2_188, %local_8_194
  %5 = icmp sge i32 %local_7_217, %4
  br i1 %5, label %bci_610, label %bci_216

bci_610:                                          ; preds = %bci_199
  ret void

bci_216:                                          ; preds = %bci_199
  br i1 %cond, label %bci_591, label %bci_221

bci_591:                                          ; preds = %bci_216
  %6 = add i32 1, %local_7_217
  br label %bci_199

bci_221:                                          ; preds = %bci_216
  %7 = srem i32 %local_7_217, 6
  %8 = add i32 %7, 114
  switch i32 %8, label %done405 [
    i32 114, label %bci_562
    i32 116, label %bci_304
    i32 117, label %bci_395
    i32 118, label %bci_407
    i32 119, label %bci_419
  ]

bci_419:                                          ; preds = %bci_221
  %9 = sub i32 %local_2_188, %local_11_197
  br label %bci_435

bci_435:                                          ; preds = %not_zero322, %bci_419
  br i1 %cond, label %not_zero265, label %never_deopt

not_zero265:                                      ; preds = %bci_435
  br i1 %cond, label %in_bounds, label %out_of_bounds

in_bounds:                                        ; preds = %not_zero265
  br i1 %cond, label %not_zero322, label %never_deopt

not_zero322:                                      ; preds = %in_bounds
  br i1 %cond, label %bci_562, label %bci_435

bci_407:                                          ; preds = %bci_221
  br label %bci_562

bci_395:                                          ; preds = %bci_221
  br i1 %cond, label %done395, label %general_case394

general_case394:                                  ; preds = %bci_395
  %10 = srem i32 %local_2_188, 0
  br label %done395

done395:                                          ; preds = %general_case394, %bci_395
  %result396 = phi i32 [ %10, %general_case394 ], [ 0, %bci_395 ]
  br label %bci_562

bci_304:                                          ; preds = %bci_221
  br i1 %cond, label %bci_371, label %bci_326

done405:                                          ; preds = %bci_221
  br i1 %cond, label %bci_562, label %done437

done437:                                          ; preds = %done405
  br label %bci_562

bci_326:                                          ; preds = %bci_304, %bci_360
  %local_7_113 = phi i32 [ %local_7_, %bci_360 ], [ %local_7_217, %bci_304 ]
  br label %bci_329

bci_337:                                          ; preds = %bci_329
  br label %bci_329

never_deopt:                                      ; preds = %in_bounds, %bci_435, %bci_371
  ret void

out_of_bounds:                                    ; preds = %not_zero265
  br i1 %cond, label %bci_565, label %not_subtype

bci_565:                                          ; preds = %out_of_bounds
  br label %bci_604

not_subtype:                                      ; preds = %out_of_bounds
  br label %bci_604
}
