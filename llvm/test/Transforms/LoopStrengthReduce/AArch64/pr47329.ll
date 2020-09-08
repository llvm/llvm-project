; RUN: opt < %s -loop-reduce
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@d = internal unnamed_addr global i32** null, align 8

define dso_local i32 @main() local_unnamed_addr {
entry:
  %.pre.pre = load i32**, i32*** @d, align 8
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %entry
  %i = phi i32** [ %.pre.pre, %entry ], [ %incdec.ptr, %for.body9 ]
  %incdec.ptr = getelementptr inbounds i32*, i32** %i, i64 -1
  br i1 undef, label %for.body9, label %for.inc

for.inc:                                          ; preds = %for.body9
  br label %for.body9.118

for.body9.1:                                      ; preds = %for.inc.547, %for.body9.1
  %i1 = phi i32** [ %incdec.ptr.1, %for.body9.1 ], [ %incdec.ptr.542, %for.inc.547 ]
  %incdec.ptr.1 = getelementptr inbounds i32*, i32** %i1, i64 -1
  br i1 undef, label %for.body9.1, label %for.inc.1

for.inc.1:                                        ; preds = %for.body9.1
  br label %for.body9.1.1

for.body9.2:                                      ; preds = %for.inc.1.5, %for.body9.2
  %i2 = phi i32** [ %incdec.ptr.2, %for.body9.2 ], [ %incdec.ptr.1.5, %for.inc.1.5 ]
  %incdec.ptr.2 = getelementptr inbounds i32*, i32** %i2, i64 -1
  br i1 undef, label %for.body9.2, label %for.inc.2

for.inc.2:                                        ; preds = %for.body9.2
  br label %for.body9.2.1

for.body9.3:                                      ; preds = %for.inc.2.5, %for.body9.3
  %i3 = phi i32** [ %incdec.ptr.3, %for.body9.3 ], [ %incdec.ptr.2.5, %for.inc.2.5 ]
  %incdec.ptr.3 = getelementptr inbounds i32*, i32** %i3, i64 -1
  br i1 undef, label %for.body9.3, label %for.inc.3

for.inc.3:                                        ; preds = %for.body9.3
  br label %for.body9.3.1

for.body9.4:                                      ; preds = %for.inc.3.5, %for.body9.4
  %i4 = phi i32** [ %incdec.ptr.4, %for.body9.4 ], [ %incdec.ptr.3.5, %for.inc.3.5 ]
  %incdec.ptr.4 = getelementptr inbounds i32*, i32** %i4, i64 -1
  br i1 undef, label %for.body9.4, label %for.inc.4

for.inc.4:                                        ; preds = %for.body9.4
  br label %for.body9.4.1

for.body9.5:                                      ; preds = %for.inc.4.5, %for.body9.5
  %i5 = phi i32** [ %incdec.ptr.5, %for.body9.5 ], [ %incdec.ptr.4.5, %for.inc.4.5 ]
  %incdec.ptr.5 = getelementptr inbounds i32*, i32** %i5, i64 -1
  br i1 undef, label %for.body9.5, label %for.inc.5

for.inc.5:                                        ; preds = %for.body9.5
  br label %for.body9.5.1

for.body9.5.1:                                    ; preds = %for.body9.5.1, %for.inc.5
  %i6 = phi i32** [ %incdec.ptr.5.1, %for.body9.5.1 ], [ %incdec.ptr.5, %for.inc.5 ]
  %incdec.ptr.5.1 = getelementptr inbounds i32*, i32** %i6, i64 -1
  br i1 undef, label %for.body9.5.1, label %for.inc.5.1

for.inc.5.1:                                      ; preds = %for.body9.5.1
  br label %for.body9.5.2

for.body9.5.2:                                    ; preds = %for.body9.5.2, %for.inc.5.1
  %i7 = phi i32** [ %incdec.ptr.5.2, %for.body9.5.2 ], [ %incdec.ptr.5.1, %for.inc.5.1 ]
  %incdec.ptr.5.2 = getelementptr inbounds i32*, i32** %i7, i64 -1
  br i1 undef, label %for.body9.5.2, label %for.inc.5.2

for.inc.5.2:                                      ; preds = %for.body9.5.2
  br label %for.body9.5.3

for.body9.5.3:                                    ; preds = %for.body9.5.3, %for.inc.5.2
  %i8 = phi i32** [ %incdec.ptr.5.3, %for.body9.5.3 ], [ %incdec.ptr.5.2, %for.inc.5.2 ]
  %incdec.ptr.5.3 = getelementptr inbounds i32*, i32** %i8, i64 -1
  br i1 undef, label %for.body9.5.3, label %for.inc.5.3

for.inc.5.3:                                      ; preds = %for.body9.5.3
  br label %for.body9.5.4

for.body9.5.4:                                    ; preds = %for.body9.5.4, %for.inc.5.3
  %i9 = phi i32** [ %incdec.ptr.5.4, %for.body9.5.4 ], [ %incdec.ptr.5.3, %for.inc.5.3 ]
  %incdec.ptr.5.4 = getelementptr inbounds i32*, i32** %i9, i64 -1
  br i1 undef, label %for.body9.5.4, label %for.inc.5.4

for.inc.5.4:                                      ; preds = %for.body9.5.4
  br label %for.body9.5.5

for.body9.5.5:                                    ; preds = %for.body9.5.5, %for.inc.5.4
  %i10 = phi i32** [ undef, %for.body9.5.5 ], [ %incdec.ptr.5.4, %for.inc.5.4 ]
  %i11 = bitcast i32** %i10 to i64*
  %i12 = load i64, i64* %i11, align 8
  br label %for.body9.5.5

for.body9.4.1:                                    ; preds = %for.body9.4.1, %for.inc.4
  %i13 = phi i32** [ %incdec.ptr.4.1, %for.body9.4.1 ], [ %incdec.ptr.4, %for.inc.4 ]
  %incdec.ptr.4.1 = getelementptr inbounds i32*, i32** %i13, i64 -1
  br i1 undef, label %for.body9.4.1, label %for.inc.4.1

for.inc.4.1:                                      ; preds = %for.body9.4.1
  br label %for.body9.4.2

for.body9.4.2:                                    ; preds = %for.body9.4.2, %for.inc.4.1
  %i14 = phi i32** [ %incdec.ptr.4.2, %for.body9.4.2 ], [ %incdec.ptr.4.1, %for.inc.4.1 ]
  %incdec.ptr.4.2 = getelementptr inbounds i32*, i32** %i14, i64 -1
  br i1 undef, label %for.body9.4.2, label %for.inc.4.2

for.inc.4.2:                                      ; preds = %for.body9.4.2
  br label %for.body9.4.3

for.body9.4.3:                                    ; preds = %for.body9.4.3, %for.inc.4.2
  %i15 = phi i32** [ %incdec.ptr.4.3, %for.body9.4.3 ], [ %incdec.ptr.4.2, %for.inc.4.2 ]
  %incdec.ptr.4.3 = getelementptr inbounds i32*, i32** %i15, i64 -1
  br i1 undef, label %for.body9.4.3, label %for.inc.4.3

for.inc.4.3:                                      ; preds = %for.body9.4.3
  br label %for.body9.4.4

for.body9.4.4:                                    ; preds = %for.body9.4.4, %for.inc.4.3
  %i16 = phi i32** [ %incdec.ptr.4.4, %for.body9.4.4 ], [ %incdec.ptr.4.3, %for.inc.4.3 ]
  %incdec.ptr.4.4 = getelementptr inbounds i32*, i32** %i16, i64 -1
  br i1 undef, label %for.body9.4.4, label %for.inc.4.4

for.inc.4.4:                                      ; preds = %for.body9.4.4
  br label %for.body9.4.5

for.body9.4.5:                                    ; preds = %for.body9.4.5, %for.inc.4.4
  %i17 = phi i32** [ %incdec.ptr.4.5, %for.body9.4.5 ], [ %incdec.ptr.4.4, %for.inc.4.4 ]
  %incdec.ptr.4.5 = getelementptr inbounds i32*, i32** %i17, i64 -1
  br i1 undef, label %for.body9.4.5, label %for.inc.4.5

for.inc.4.5:                                      ; preds = %for.body9.4.5
  br label %for.body9.5

for.body9.3.1:                                    ; preds = %for.body9.3.1, %for.inc.3
  %i18 = phi i32** [ %incdec.ptr.3.1, %for.body9.3.1 ], [ %incdec.ptr.3, %for.inc.3 ]
  %incdec.ptr.3.1 = getelementptr inbounds i32*, i32** %i18, i64 -1
  br i1 undef, label %for.body9.3.1, label %for.inc.3.1

for.inc.3.1:                                      ; preds = %for.body9.3.1
  br label %for.body9.3.2

for.body9.3.2:                                    ; preds = %for.body9.3.2, %for.inc.3.1
  %i19 = phi i32** [ %incdec.ptr.3.2, %for.body9.3.2 ], [ %incdec.ptr.3.1, %for.inc.3.1 ]
  %incdec.ptr.3.2 = getelementptr inbounds i32*, i32** %i19, i64 -1
  br i1 undef, label %for.body9.3.2, label %for.inc.3.2

for.inc.3.2:                                      ; preds = %for.body9.3.2
  br label %for.body9.3.3

for.body9.3.3:                                    ; preds = %for.body9.3.3, %for.inc.3.2
  %i20 = phi i32** [ %incdec.ptr.3.3, %for.body9.3.3 ], [ %incdec.ptr.3.2, %for.inc.3.2 ]
  %incdec.ptr.3.3 = getelementptr inbounds i32*, i32** %i20, i64 -1
  br i1 undef, label %for.body9.3.3, label %for.inc.3.3

for.inc.3.3:                                      ; preds = %for.body9.3.3
  br label %for.body9.3.4

for.body9.3.4:                                    ; preds = %for.body9.3.4, %for.inc.3.3
  %i21 = phi i32** [ %incdec.ptr.3.4, %for.body9.3.4 ], [ %incdec.ptr.3.3, %for.inc.3.3 ]
  %incdec.ptr.3.4 = getelementptr inbounds i32*, i32** %i21, i64 -1
  br i1 undef, label %for.body9.3.4, label %for.inc.3.4

for.inc.3.4:                                      ; preds = %for.body9.3.4
  br label %for.body9.3.5

for.body9.3.5:                                    ; preds = %for.body9.3.5, %for.inc.3.4
  %i22 = phi i32** [ %incdec.ptr.3.5, %for.body9.3.5 ], [ %incdec.ptr.3.4, %for.inc.3.4 ]
  %incdec.ptr.3.5 = getelementptr inbounds i32*, i32** %i22, i64 -1
  br i1 undef, label %for.body9.3.5, label %for.inc.3.5

for.inc.3.5:                                      ; preds = %for.body9.3.5
  br label %for.body9.4

for.body9.2.1:                                    ; preds = %for.body9.2.1, %for.inc.2
  %i23 = phi i32** [ %incdec.ptr.2.1, %for.body9.2.1 ], [ %incdec.ptr.2, %for.inc.2 ]
  %incdec.ptr.2.1 = getelementptr inbounds i32*, i32** %i23, i64 -1
  br i1 undef, label %for.body9.2.1, label %for.inc.2.1

for.inc.2.1:                                      ; preds = %for.body9.2.1
  br label %for.body9.2.2

for.body9.2.2:                                    ; preds = %for.body9.2.2, %for.inc.2.1
  %i24 = phi i32** [ %incdec.ptr.2.2, %for.body9.2.2 ], [ %incdec.ptr.2.1, %for.inc.2.1 ]
  %incdec.ptr.2.2 = getelementptr inbounds i32*, i32** %i24, i64 -1
  br i1 undef, label %for.body9.2.2, label %for.inc.2.2

for.inc.2.2:                                      ; preds = %for.body9.2.2
  br label %for.body9.2.3

for.body9.2.3:                                    ; preds = %for.body9.2.3, %for.inc.2.2
  %i25 = phi i32** [ %incdec.ptr.2.3, %for.body9.2.3 ], [ %incdec.ptr.2.2, %for.inc.2.2 ]
  %incdec.ptr.2.3 = getelementptr inbounds i32*, i32** %i25, i64 -1
  br i1 undef, label %for.body9.2.3, label %for.inc.2.3

for.inc.2.3:                                      ; preds = %for.body9.2.3
  br label %for.body9.2.4

for.body9.2.4:                                    ; preds = %for.body9.2.4, %for.inc.2.3
  %i26 = phi i32** [ %incdec.ptr.2.4, %for.body9.2.4 ], [ %incdec.ptr.2.3, %for.inc.2.3 ]
  %incdec.ptr.2.4 = getelementptr inbounds i32*, i32** %i26, i64 -1
  br i1 undef, label %for.body9.2.4, label %for.inc.2.4

for.inc.2.4:                                      ; preds = %for.body9.2.4
  br label %for.body9.2.5

for.body9.2.5:                                    ; preds = %for.body9.2.5, %for.inc.2.4
  %i27 = phi i32** [ %incdec.ptr.2.5, %for.body9.2.5 ], [ %incdec.ptr.2.4, %for.inc.2.4 ]
  %incdec.ptr.2.5 = getelementptr inbounds i32*, i32** %i27, i64 -1
  br i1 undef, label %for.body9.2.5, label %for.inc.2.5

for.inc.2.5:                                      ; preds = %for.body9.2.5
  br label %for.body9.3

for.body9.1.1:                                    ; preds = %for.body9.1.1, %for.inc.1
  %i28 = phi i32** [ %incdec.ptr.1.1, %for.body9.1.1 ], [ %incdec.ptr.1, %for.inc.1 ]
  %incdec.ptr.1.1 = getelementptr inbounds i32*, i32** %i28, i64 -1
  br i1 undef, label %for.body9.1.1, label %for.inc.1.1

for.inc.1.1:                                      ; preds = %for.body9.1.1
  br label %for.body9.1.2

for.body9.1.2:                                    ; preds = %for.body9.1.2, %for.inc.1.1
  %i29 = phi i32** [ %incdec.ptr.1.2, %for.body9.1.2 ], [ %incdec.ptr.1.1, %for.inc.1.1 ]
  %incdec.ptr.1.2 = getelementptr inbounds i32*, i32** %i29, i64 -1
  br i1 undef, label %for.body9.1.2, label %for.inc.1.2

for.inc.1.2:                                      ; preds = %for.body9.1.2
  br label %for.body9.1.3

for.body9.1.3:                                    ; preds = %for.body9.1.3, %for.inc.1.2
  %i30 = phi i32** [ %incdec.ptr.1.3, %for.body9.1.3 ], [ %incdec.ptr.1.2, %for.inc.1.2 ]
  %incdec.ptr.1.3 = getelementptr inbounds i32*, i32** %i30, i64 -1
  br i1 undef, label %for.body9.1.3, label %for.inc.1.3

for.inc.1.3:                                      ; preds = %for.body9.1.3
  br label %for.body9.1.4

for.body9.1.4:                                    ; preds = %for.body9.1.4, %for.inc.1.3
  %i31 = phi i32** [ %incdec.ptr.1.4, %for.body9.1.4 ], [ %incdec.ptr.1.3, %for.inc.1.3 ]
  %incdec.ptr.1.4 = getelementptr inbounds i32*, i32** %i31, i64 -1
  br i1 undef, label %for.body9.1.4, label %for.inc.1.4

for.inc.1.4:                                      ; preds = %for.body9.1.4
  br label %for.body9.1.5

for.body9.1.5:                                    ; preds = %for.body9.1.5, %for.inc.1.4
  %i32 = phi i32** [ %incdec.ptr.1.5, %for.body9.1.5 ], [ %incdec.ptr.1.4, %for.inc.1.4 ]
  %incdec.ptr.1.5 = getelementptr inbounds i32*, i32** %i32, i64 -1
  br i1 undef, label %for.body9.1.5, label %for.inc.1.5

for.inc.1.5:                                      ; preds = %for.body9.1.5
  br label %for.body9.2

for.body9.118:                                    ; preds = %for.body9.118, %for.inc
  %i33 = phi i32** [ %incdec.ptr, %for.inc ], [ %incdec.ptr.114, %for.body9.118 ]
  %incdec.ptr.114 = getelementptr inbounds i32*, i32** %i33, i64 -1
  br i1 undef, label %for.body9.118, label %for.inc.119

for.inc.119:                                      ; preds = %for.body9.118
  br label %for.body9.225

for.body9.225:                                    ; preds = %for.body9.225, %for.inc.119
  %i34 = phi i32** [ %incdec.ptr.114, %for.inc.119 ], [ %incdec.ptr.221, %for.body9.225 ]
  %incdec.ptr.221 = getelementptr inbounds i32*, i32** %i34, i64 -1
  %i35 = bitcast i32** %i34 to i64*
  %i36 = load i64, i64* %i35, align 8
  br i1 undef, label %for.body9.225, label %for.inc.226

for.inc.226:                                      ; preds = %for.body9.225
  br label %for.body9.332

for.body9.332:                                    ; preds = %for.body9.332, %for.inc.226
  %i37 = phi i32** [ %incdec.ptr.221, %for.inc.226 ], [ %incdec.ptr.328, %for.body9.332 ]
  %incdec.ptr.328 = getelementptr inbounds i32*, i32** %i37, i64 -1
  br i1 undef, label %for.body9.332, label %for.inc.333

for.inc.333:                                      ; preds = %for.body9.332
  br label %for.body9.439

for.body9.439:                                    ; preds = %for.body9.439, %for.inc.333
  %i38 = phi i32** [ %incdec.ptr.328, %for.inc.333 ], [ %incdec.ptr.435, %for.body9.439 ]
  %incdec.ptr.435 = getelementptr inbounds i32*, i32** %i38, i64 -1
  br i1 undef, label %for.body9.439, label %for.inc.440

for.inc.440:                                      ; preds = %for.body9.439
  br label %for.body9.546

for.body9.546:                                    ; preds = %for.body9.546, %for.inc.440
  %i39 = phi i32** [ %incdec.ptr.435, %for.inc.440 ], [ %incdec.ptr.542, %for.body9.546 ]
  %incdec.ptr.542 = getelementptr inbounds i32*, i32** %i39, i64 -1
  br i1 undef, label %for.body9.546, label %for.inc.547

for.inc.547:                                      ; preds = %for.body9.546
  br label %for.body9.1
}
