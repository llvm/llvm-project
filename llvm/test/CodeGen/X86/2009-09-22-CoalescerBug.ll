; RUN: llc < %s -mtriple=x86_64-apple-darwin10

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  br i1 undef, label %bb, label %bb1

bb:                                               ; preds = %entry
  ret i32 3

bb1:                                              ; preds = %entry
  br i1 undef, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  ret i32 3

bb3:                                              ; preds = %bb1
  br i1 undef, label %bb.i18, label %quantum_getwidth.exit

bb.i18:                                           ; preds = %bb.i18, %bb3
  br i1 undef, label %bb.i18, label %quantum_getwidth.exit

quantum_getwidth.exit:                            ; preds = %bb.i18, %bb3
  br i1 undef, label %bb4, label %bb6.preheader

bb4:                                              ; preds = %quantum_getwidth.exit
  unreachable

bb6.preheader:                                    ; preds = %quantum_getwidth.exit
  br i1 undef, label %bb.i1, label %bb1.i2

bb.i1:                                            ; preds = %bb6.preheader
  unreachable

bb1.i2:                                           ; preds = %bb6.preheader
  br i1 undef, label %bb2.i, label %bb3.i4

bb2.i:                                            ; preds = %bb1.i2
  unreachable

bb3.i4:                                           ; preds = %bb1.i2
  br i1 undef, label %quantum_new_qureg.exit, label %bb4.i

bb4.i:                                            ; preds = %bb3.i4
  unreachable

quantum_new_qureg.exit:                           ; preds = %bb3.i4
  br i1 undef, label %bb9, label %bb11.thread

bb11.thread:                                      ; preds = %quantum_new_qureg.exit
  %.cast.i = zext i32 undef to i64                ; <i64> [#uses=1]
  br label %bb.i37

bb9:                                              ; preds = %quantum_new_qureg.exit
  unreachable

bb.i37:                                           ; preds = %bb.i37, %bb11.thread
  %0 = load i64* undef, align 8                   ; <i64> [#uses=1]
  %1 = shl i64 %0, %.cast.i                       ; <i64> [#uses=1]
  store i64 %1, i64* undef, align 8
  br i1 undef, label %bb.i37, label %quantum_addscratch.exit

quantum_addscratch.exit:                          ; preds = %bb.i37
  br i1 undef, label %bb12.preheader, label %bb14

bb12.preheader:                                   ; preds = %quantum_addscratch.exit
  unreachable

bb14:                                             ; preds = %quantum_addscratch.exit
  br i1 undef, label %bb17, label %bb.nph

bb.nph:                                           ; preds = %bb14
  unreachable

bb17:                                             ; preds = %bb14
  br i1 undef, label %bb1.i7, label %quantum_measure.exit

bb1.i7:                                           ; preds = %bb17
  br label %quantum_measure.exit

quantum_measure.exit:                             ; preds = %bb1.i7, %bb17
  switch i32 undef, label %bb21 [
    i32 -1, label %bb18
    i32 0, label %bb20
  ]

bb18:                                             ; preds = %quantum_measure.exit
  unreachable

bb20:                                             ; preds = %quantum_measure.exit
  unreachable

bb21:                                             ; preds = %quantum_measure.exit
  br i1 undef, label %quantum_frac_approx.exit, label %bb1.i

bb1.i:                                            ; preds = %bb21
  unreachable

quantum_frac_approx.exit:                         ; preds = %bb21
  br i1 undef, label %bb25, label %bb26

bb25:                                             ; preds = %quantum_frac_approx.exit
  unreachable

bb26:                                             ; preds = %quantum_frac_approx.exit
  br i1 undef, label %quantum_gcd.exit, label %bb.i

bb.i:                                             ; preds = %bb.i, %bb26
  br i1 undef, label %quantum_gcd.exit, label %bb.i

quantum_gcd.exit:                                 ; preds = %bb.i, %bb26
  br i1 undef, label %bb32, label %bb33

bb32:                                             ; preds = %quantum_gcd.exit
  br i1 undef, label %bb.i.i, label %quantum_delete_qureg.exit

bb.i.i:                                           ; preds = %bb32
  ret i32 0

quantum_delete_qureg.exit:                        ; preds = %bb32
  ret i32 0

bb33:                                             ; preds = %quantum_gcd.exit
  unreachable
}
