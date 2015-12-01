; RUN: llc %s -enable-shrink-wrap=true -o - | FileCheck %s

target triple = "thumbv6m-none-none-eabi"

@retval = global i32 0, align 4

define i32 @test(i32 %i, i32 %argc, i8** nocapture readonly %argv) {
  %1 = icmp sgt i32 %argc, %i
  br i1 %1, label %2, label %19

  %3 = getelementptr inbounds i8*, i8** %argv, i32 %i
  %4 = load i8*, i8** %3, align 4
  %5 = load i8, i8* %4, align 1
  %6 = icmp eq i8 %5, 45
  %7 = getelementptr inbounds i8, i8* %4, i32 1
  %. = select i1 %6, i8* %7, i8* %4
  %.1 = select i1 %6, i32 -1, i32 1
  %8 = load i8, i8* %., align 1
  %.off2 = add i8 %8, -48
  %9 = icmp ult i8 %.off2, 10
  %.pre = load i32, i32* @retval, align 4
  br i1 %9, label %.lr.ph.preheader, label %.critedge

.lr.ph.preheader:                                 ; preds = %2
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.lr.ph
  %10 = phi i32 [ %14, %.lr.ph ], [ %.pre, %.lr.ph.preheader ]
  %11 = phi i8 [ %15, %.lr.ph ], [ %8, %.lr.ph.preheader ]
  %valstring.03 = phi i8* [ %13, %.lr.ph ], [ %., %.lr.ph.preheader ]
  %12 = zext i8 %11 to i32
  %13 = getelementptr inbounds i8, i8* %valstring.03, i32 1
  %14 = add nsw i32 %10, %12
  store i32 %14, i32* @retval, align 4
  %15 = load i8, i8* %13, align 1
  %.off = add i8 %15, -48
  %16 = icmp ult i8 %.off, 10
  br i1 %16, label %.lr.ph, label %.critedge.loopexit

.critedge.loopexit:                               ; preds = %.lr.ph
  %.lcssa = phi i32 [ %14, %.lr.ph ]
  br label %.critedge

.critedge:                                        ; preds = %.critedge.loopexit, %2
  %17 = phi i32 [ %.pre, %2 ], [ %.lcssa, %.critedge.loopexit ]
  %18 = mul nsw i32 %17, %.1
  store i32 %18, i32* @retval, align 4
  br label %19

; <label>:19                                      ; preds = %.critedge, %0
  ret i32 0
}

; CHECK: push {r4, r5, r7, lr}
; CHECK: pop {r4, r5, r7}
; CHECK: pop {r0}
; CHECK: mov lr, r0
; CHECK: movs r0, #0
; CHECK: bx  lr

