; RUN: opt -passes=loop-reduce -S < %s | FileCheck %s
; REQUIRES: x86-registered-target
;
; A sequence of loops whose recurrence starts depend on earlier loops.
; Proving comparisons of the starts must not recursively branch into full
; induction proofs from the nonrecursive predicate prover (issue #221391).
; Calls are to a declared function. No poison pointers, null calls,
; unchecked memory accesses, or wrapping arithmetic are involved.
target triple = "x86_64-unknown-linux-gnu"
declare i1 @keep_going(i64)
define i64 @chain() {
; CHECK-LABEL: define i64 @chain()
; CHECK: loop0:
; CHECK: call i1 @keep_going
; CHECK: loop16:
; CHECK: call i1 @keep_going
; CHECK: ret i64
entry:
  br label %loop0
loop0:
  %iv0 = phi i64 [0, %entry], [%next0, %cont0]
  %next0 = add nuw nsw i64 %iv0, 1
  %go0 = call i1 @keep_going(i64 %iv0)
  br i1 %go0, label %cont0, label %after0
cont0:
  %guard0 = icmp ult i64 %iv0, 65533
  br i1 %guard0, label %loop0, label %exit
after0:
  %start1 = add nuw nsw i64 %iv0, 2
  br label %loop1
loop1:
  %iv1 = phi i64 [%start1, %after0], [%next1, %cont1]
  %next1 = add nuw nsw i64 %iv1, 1
  %go1 = call i1 @keep_going(i64 %iv1)
  br i1 %go1, label %cont1, label %after1
cont1:
  %guard1 = icmp ult i64 %iv1, 65533
  br i1 %guard1, label %loop1, label %exit
after1:
  %start2 = add nuw nsw i64 %iv1, 2
  br label %loop2
loop2:
  %iv2 = phi i64 [%start2, %after1], [%next2, %cont2]
  %next2 = add nuw nsw i64 %iv2, 1
  %go2 = call i1 @keep_going(i64 %iv2)
  br i1 %go2, label %cont2, label %after2
cont2:
  %guard2 = icmp ult i64 %iv2, 65533
  br i1 %guard2, label %loop2, label %exit
after2:
  %start3 = add nuw nsw i64 %iv2, 2
  br label %loop3
loop3:
  %iv3 = phi i64 [%start3, %after2], [%next3, %cont3]
  %next3 = add nuw nsw i64 %iv3, 1
  %go3 = call i1 @keep_going(i64 %iv3)
  br i1 %go3, label %cont3, label %after3
cont3:
  %guard3 = icmp ult i64 %iv3, 65533
  br i1 %guard3, label %loop3, label %exit
after3:
  %start4 = add nuw nsw i64 %iv3, 2
  br label %loop4
loop4:
  %iv4 = phi i64 [%start4, %after3], [%next4, %cont4]
  %next4 = add nuw nsw i64 %iv4, 1
  %go4 = call i1 @keep_going(i64 %iv4)
  br i1 %go4, label %cont4, label %after4
cont4:
  %guard4 = icmp ult i64 %iv4, 65533
  br i1 %guard4, label %loop4, label %exit
after4:
  %start5 = add nuw nsw i64 %iv4, 2
  br label %loop5
loop5:
  %iv5 = phi i64 [%start5, %after4], [%next5, %cont5]
  %next5 = add nuw nsw i64 %iv5, 1
  %go5 = call i1 @keep_going(i64 %iv5)
  br i1 %go5, label %cont5, label %after5
cont5:
  %guard5 = icmp ult i64 %iv5, 65533
  br i1 %guard5, label %loop5, label %exit
after5:
  %start6 = add nuw nsw i64 %iv5, 2
  br label %loop6
loop6:
  %iv6 = phi i64 [%start6, %after5], [%next6, %cont6]
  %next6 = add nuw nsw i64 %iv6, 1
  %go6 = call i1 @keep_going(i64 %iv6)
  br i1 %go6, label %cont6, label %after6
cont6:
  %guard6 = icmp ult i64 %iv6, 65533
  br i1 %guard6, label %loop6, label %exit
after6:
  %start7 = add nuw nsw i64 %iv6, 2
  br label %loop7
loop7:
  %iv7 = phi i64 [%start7, %after6], [%next7, %cont7]
  %next7 = add nuw nsw i64 %iv7, 1
  %go7 = call i1 @keep_going(i64 %iv7)
  br i1 %go7, label %cont7, label %after7
cont7:
  %guard7 = icmp ult i64 %iv7, 65533
  br i1 %guard7, label %loop7, label %exit
after7:
  %start8 = add nuw nsw i64 %iv7, 2
  br label %loop8
loop8:
  %iv8 = phi i64 [%start8, %after7], [%next8, %cont8]
  %next8 = add nuw nsw i64 %iv8, 1
  %go8 = call i1 @keep_going(i64 %iv8)
  br i1 %go8, label %cont8, label %after8
cont8:
  %guard8 = icmp ult i64 %iv8, 65533
  br i1 %guard8, label %loop8, label %exit
after8:
  %start9 = add nuw nsw i64 %iv8, 2
  br label %loop9
loop9:
  %iv9 = phi i64 [%start9, %after8], [%next9, %cont9]
  %next9 = add nuw nsw i64 %iv9, 1
  %go9 = call i1 @keep_going(i64 %iv9)
  br i1 %go9, label %cont9, label %after9
cont9:
  %guard9 = icmp ult i64 %iv9, 65533
  br i1 %guard9, label %loop9, label %exit
after9:
  %start10 = add nuw nsw i64 %iv9, 2
  br label %loop10
loop10:
  %iv10 = phi i64 [%start10, %after9], [%next10, %cont10]
  %next10 = add nuw nsw i64 %iv10, 1
  %go10 = call i1 @keep_going(i64 %iv10)
  br i1 %go10, label %cont10, label %after10
cont10:
  %guard10 = icmp ult i64 %iv10, 65533
  br i1 %guard10, label %loop10, label %exit
after10:
  %start11 = add nuw nsw i64 %iv10, 2
  br label %loop11
loop11:
  %iv11 = phi i64 [%start11, %after10], [%next11, %cont11]
  %next11 = add nuw nsw i64 %iv11, 1
  %go11 = call i1 @keep_going(i64 %iv11)
  br i1 %go11, label %cont11, label %after11
cont11:
  %guard11 = icmp ult i64 %iv11, 65533
  br i1 %guard11, label %loop11, label %exit
after11:
  %start12 = add nuw nsw i64 %iv11, 2
  br label %loop12
loop12:
  %iv12 = phi i64 [%start12, %after11], [%next12, %cont12]
  %next12 = add nuw nsw i64 %iv12, 1
  %go12 = call i1 @keep_going(i64 %iv12)
  br i1 %go12, label %cont12, label %after12
cont12:
  %guard12 = icmp ult i64 %iv12, 65533
  br i1 %guard12, label %loop12, label %exit
after12:
  %start13 = add nuw nsw i64 %iv12, 2
  br label %loop13
loop13:
  %iv13 = phi i64 [%start13, %after12], [%next13, %cont13]
  %next13 = add nuw nsw i64 %iv13, 1
  %go13 = call i1 @keep_going(i64 %iv13)
  br i1 %go13, label %cont13, label %after13
cont13:
  %guard13 = icmp ult i64 %iv13, 65533
  br i1 %guard13, label %loop13, label %exit
after13:
  %start14 = add nuw nsw i64 %iv13, 2
  br label %loop14
loop14:
  %iv14 = phi i64 [%start14, %after13], [%next14, %cont14]
  %next14 = add nuw nsw i64 %iv14, 1
  %go14 = call i1 @keep_going(i64 %iv14)
  br i1 %go14, label %cont14, label %after14
cont14:
  %guard14 = icmp ult i64 %iv14, 65533
  br i1 %guard14, label %loop14, label %exit
after14:
  %start15 = add nuw nsw i64 %iv14, 2
  br label %loop15
loop15:
  %iv15 = phi i64 [%start15, %after14], [%next15, %cont15]
  %next15 = add nuw nsw i64 %iv15, 1
  %go15 = call i1 @keep_going(i64 %iv15)
  br i1 %go15, label %cont15, label %after15
cont15:
  %guard15 = icmp ult i64 %iv15, 65533
  br i1 %guard15, label %loop15, label %exit
after15:
  %start16 = add nuw nsw i64 %iv15, 2
  br label %loop16
loop16:
  %iv16 = phi i64 [%start16, %after15], [%next16, %cont16]
  %next16 = add nuw nsw i64 %iv16, 1
  %go16 = call i1 @keep_going(i64 %iv16)
  br i1 %go16, label %cont16, label %after16
cont16:
  %guard16 = icmp ult i64 %iv16, 65533
  br i1 %guard16, label %loop16, label %exit
after16:
  ret i64 %iv16
exit:
  ret i64 65534
}
