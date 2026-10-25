; RUN: opt -passes=loop-reduce -S < %s | FileCheck %s

; Regression test for compile-time blowup when comparing recurrence starts
; that depend on preceding loops (PR221391).

target triple = "x86_64-unknown-linux-gnu"

declare i1 @keep_going(i64)

define i64 @chain() {
; CHECK-LABEL: define i64 @chain()
; CHECK: loop0.header:
; CHECK: call i1 @keep_going
; CHECK: loop16.header:
; CHECK: call i1 @keep_going
; CHECK: ret i64

entry:
  br label %loop0.header

loop0.header:
  %iv0 = phi i64 [ 0, %entry ], [ %next0, %loop0.latch ]
  %next0 = add nuw nsw i64 %iv0, 1
  %go0 = call i1 @keep_going(i64 %iv0)
  br i1 %go0, label %loop0.latch, label %loop0.exit

loop0.latch:
  %guard0 = icmp ult i64 %iv0, 65533
  br i1 %guard0, label %loop0.header, label %exit

loop0.exit:
  %start1 = add nuw nsw i64 %iv0, 2
  br label %loop1.header

loop1.header:
  %iv1 = phi i64 [ %start1, %loop0.exit ], [ %next1, %loop1.latch ]
  %next1 = add nuw nsw i64 %iv1, 1
  %go1 = call i1 @keep_going(i64 %iv1)
  br i1 %go1, label %loop1.latch, label %loop1.exit

loop1.latch:
  %guard1 = icmp ult i64 %iv1, 65533
  br i1 %guard1, label %loop1.header, label %exit

loop1.exit:
  %start2 = add nuw nsw i64 %iv1, 2
  br label %loop2.header

loop2.header:
  %iv2 = phi i64 [ %start2, %loop1.exit ], [ %next2, %loop2.latch ]
  %next2 = add nuw nsw i64 %iv2, 1
  %go2 = call i1 @keep_going(i64 %iv2)
  br i1 %go2, label %loop2.latch, label %loop2.exit

loop2.latch:
  %guard2 = icmp ult i64 %iv2, 65533
  br i1 %guard2, label %loop2.header, label %exit

loop2.exit:
  %start3 = add nuw nsw i64 %iv2, 2
  br label %loop3.header

loop3.header:
  %iv3 = phi i64 [ %start3, %loop2.exit ], [ %next3, %loop3.latch ]
  %next3 = add nuw nsw i64 %iv3, 1
  %go3 = call i1 @keep_going(i64 %iv3)
  br i1 %go3, label %loop3.latch, label %loop3.exit

loop3.latch:
  %guard3 = icmp ult i64 %iv3, 65533
  br i1 %guard3, label %loop3.header, label %exit

loop3.exit:
  %start4 = add nuw nsw i64 %iv3, 2
  br label %loop4.header

loop4.header:
  %iv4 = phi i64 [ %start4, %loop3.exit ], [ %next4, %loop4.latch ]
  %next4 = add nuw nsw i64 %iv4, 1
  %go4 = call i1 @keep_going(i64 %iv4)
  br i1 %go4, label %loop4.latch, label %loop4.exit

loop4.latch:
  %guard4 = icmp ult i64 %iv4, 65533
  br i1 %guard4, label %loop4.header, label %exit

loop4.exit:
  %start5 = add nuw nsw i64 %iv4, 2
  br label %loop5.header

loop5.header:
  %iv5 = phi i64 [ %start5, %loop4.exit ], [ %next5, %loop5.latch ]
  %next5 = add nuw nsw i64 %iv5, 1
  %go5 = call i1 @keep_going(i64 %iv5)
  br i1 %go5, label %loop5.latch, label %loop5.exit

loop5.latch:
  %guard5 = icmp ult i64 %iv5, 65533
  br i1 %guard5, label %loop5.header, label %exit

loop5.exit:
  %start6 = add nuw nsw i64 %iv5, 2
  br label %loop6.header

loop6.header:
  %iv6 = phi i64 [ %start6, %loop5.exit ], [ %next6, %loop6.latch ]
  %next6 = add nuw nsw i64 %iv6, 1
  %go6 = call i1 @keep_going(i64 %iv6)
  br i1 %go6, label %loop6.latch, label %loop6.exit

loop6.latch:
  %guard6 = icmp ult i64 %iv6, 65533
  br i1 %guard6, label %loop6.header, label %exit

loop6.exit:
  %start7 = add nuw nsw i64 %iv6, 2
  br label %loop7.header

loop7.header:
  %iv7 = phi i64 [ %start7, %loop6.exit ], [ %next7, %loop7.latch ]
  %next7 = add nuw nsw i64 %iv7, 1
  %go7 = call i1 @keep_going(i64 %iv7)
  br i1 %go7, label %loop7.latch, label %loop7.exit

loop7.latch:
  %guard7 = icmp ult i64 %iv7, 65533
  br i1 %guard7, label %loop7.header, label %exit

loop7.exit:
  %start8 = add nuw nsw i64 %iv7, 2
  br label %loop8.header

loop8.header:
  %iv8 = phi i64 [ %start8, %loop7.exit ], [ %next8, %loop8.latch ]
  %next8 = add nuw nsw i64 %iv8, 1
  %go8 = call i1 @keep_going(i64 %iv8)
  br i1 %go8, label %loop8.latch, label %loop8.exit

loop8.latch:
  %guard8 = icmp ult i64 %iv8, 65533
  br i1 %guard8, label %loop8.header, label %exit

loop8.exit:
  %start9 = add nuw nsw i64 %iv8, 2
  br label %loop9.header

loop9.header:
  %iv9 = phi i64 [ %start9, %loop8.exit ], [ %next9, %loop9.latch ]
  %next9 = add nuw nsw i64 %iv9, 1
  %go9 = call i1 @keep_going(i64 %iv9)
  br i1 %go9, label %loop9.latch, label %loop9.exit

loop9.latch:
  %guard9 = icmp ult i64 %iv9, 65533
  br i1 %guard9, label %loop9.header, label %exit

loop9.exit:
  %start10 = add nuw nsw i64 %iv9, 2
  br label %loop10.header

loop10.header:
  %iv10 = phi i64 [ %start10, %loop9.exit ], [ %next10, %loop10.latch ]
  %next10 = add nuw nsw i64 %iv10, 1
  %go10 = call i1 @keep_going(i64 %iv10)
  br i1 %go10, label %loop10.latch, label %loop10.exit

loop10.latch:
  %guard10 = icmp ult i64 %iv10, 65533
  br i1 %guard10, label %loop10.header, label %exit

loop10.exit:
  %start11 = add nuw nsw i64 %iv10, 2
  br label %loop11.header

loop11.header:
  %iv11 = phi i64 [ %start11, %loop10.exit ], [ %next11, %loop11.latch ]
  %next11 = add nuw nsw i64 %iv11, 1
  %go11 = call i1 @keep_going(i64 %iv11)
  br i1 %go11, label %loop11.latch, label %loop11.exit

loop11.latch:
  %guard11 = icmp ult i64 %iv11, 65533
  br i1 %guard11, label %loop11.header, label %exit

loop11.exit:
  %start12 = add nuw nsw i64 %iv11, 2
  br label %loop12.header

loop12.header:
  %iv12 = phi i64 [ %start12, %loop11.exit ], [ %next12, %loop12.latch ]
  %next12 = add nuw nsw i64 %iv12, 1
  %go12 = call i1 @keep_going(i64 %iv12)
  br i1 %go12, label %loop12.latch, label %loop12.exit

loop12.latch:
  %guard12 = icmp ult i64 %iv12, 65533
  br i1 %guard12, label %loop12.header, label %exit

loop12.exit:
  %start13 = add nuw nsw i64 %iv12, 2
  br label %loop13.header

loop13.header:
  %iv13 = phi i64 [ %start13, %loop12.exit ], [ %next13, %loop13.latch ]
  %next13 = add nuw nsw i64 %iv13, 1
  %go13 = call i1 @keep_going(i64 %iv13)
  br i1 %go13, label %loop13.latch, label %loop13.exit

loop13.latch:
  %guard13 = icmp ult i64 %iv13, 65533
  br i1 %guard13, label %loop13.header, label %exit

loop13.exit:
  %start14 = add nuw nsw i64 %iv13, 2
  br label %loop14.header

loop14.header:
  %iv14 = phi i64 [ %start14, %loop13.exit ], [ %next14, %loop14.latch ]
  %next14 = add nuw nsw i64 %iv14, 1
  %go14 = call i1 @keep_going(i64 %iv14)
  br i1 %go14, label %loop14.latch, label %loop14.exit

loop14.latch:
  %guard14 = icmp ult i64 %iv14, 65533
  br i1 %guard14, label %loop14.header, label %exit

loop14.exit:
  %start15 = add nuw nsw i64 %iv14, 2
  br label %loop15.header

loop15.header:
  %iv15 = phi i64 [ %start15, %loop14.exit ], [ %next15, %loop15.latch ]
  %next15 = add nuw nsw i64 %iv15, 1
  %go15 = call i1 @keep_going(i64 %iv15)
  br i1 %go15, label %loop15.latch, label %loop15.exit

loop15.latch:
  %guard15 = icmp ult i64 %iv15, 65533
  br i1 %guard15, label %loop15.header, label %exit

loop15.exit:
  %start16 = add nuw nsw i64 %iv15, 2
  br label %loop16.header

loop16.header:
  %iv16 = phi i64 [ %start16, %loop15.exit ], [ %next16, %loop16.latch ]
  %next16 = add nuw nsw i64 %iv16, 1
  %go16 = call i1 @keep_going(i64 %iv16)
  br i1 %go16, label %loop16.latch, label %loop16.exit

loop16.latch:
  %guard16 = icmp ult i64 %iv16, 65533
  br i1 %guard16, label %loop16.header, label %exit

loop16.exit:
  ret i64 %iv16

exit:
  ret i64 65534
}
