; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S -enable-if-conversion | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@A = global [36 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35], align 16
@B = global [36 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35], align 16

;CHECK-LABEL:@_Z5test1v(
;CHECK: load <4 x i32>
;CHECK: shufflevector <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret
define i32 @_Z5test1v() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %0, %1
  %p.02 = phi i32* [ getelementptr inbounds ([36 x i32], [36 x i32]* @A, i64 0, i64 18), %0 ], [ %4, %1 ]
  %b.01 = phi i32* [ getelementptr inbounds ([36 x i32], [36 x i32]* @B, i64 0, i64 0), %0 ], [ %5, %1 ]
  %2 = load i32, i32* %b.01, align 4
  %3 = shl nsw i32 %2, 1
  store i32 %3, i32* %p.02, align 4
  %4 = getelementptr inbounds i32, i32* %p.02, i64 -1
  %5 = getelementptr inbounds i32, i32* %b.01, i64 1
  %6 = icmp eq i32* %4, getelementptr ([36 x i32], [36 x i32]* @A, i64 128102389400760775, i64 3)
  br i1 %6, label %7, label %1

; <label>:7                                       ; preds = %1
  ret i32 0
}

;CHECK-LABEL: @_Z5test2v(
;CHECK: load <4 x i32>
;CHECK: shufflevector <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret
define i32 @_Z5test2v() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %0, %1
  %p.02 = phi i32* [ getelementptr inbounds ([36 x i32], [36 x i32]* @A, i64 0, i64 25), %0 ], [ %3, %1 ]
  %b.01 = phi i32* [ getelementptr inbounds ([36 x i32], [36 x i32]* @B, i64 0, i64 2), %0 ], [ %4, %1 ]
  %2 = load i32, i32* %b.01, align 4
  store i32 %2, i32* %p.02, align 4
  %3 = getelementptr inbounds i32, i32* %p.02, i64 -1
  %4 = getelementptr inbounds i32, i32* %b.01, i64 1
  %5 = icmp eq i32* %4, getelementptr inbounds ([36 x i32], [36 x i32]* @A, i64 0, i64 18)
  br i1 %5, label %6, label %1

; <label>:6                                       ; preds = %1
  ret i32 0
}

;CHECK:_Z5test3v
;CHECK: load <4 x i32>
;CHECK: shufflevector <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret
define i32 @_Z5test3v() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %0, %1
  %p.02 = phi i32* [ getelementptr inbounds ([36 x i32], [36 x i32]* @A, i64 0, i64 29), %0 ], [ %3, %1 ]
  %b.01 = phi i32* [ getelementptr inbounds ([36 x i32], [36 x i32]* @B, i64 0, i64 5), %0 ], [ %4, %1 ]
  %2 = load i32, i32* %b.01, align 4
  store i32 %2, i32* %p.02, align 4
  %3 = getelementptr inbounds i32, i32* %p.02, i64 -1
  %4 = getelementptr inbounds i32, i32* %b.01, i64 1
  %5 = icmp eq i32* %3, getelementptr ([36 x i32], [36 x i32]* @A, i64 128102389400760775, i64 3)
  br i1 %5, label %6, label %1

; <label>:6                                       ; preds = %1
  ret i32 0
}
