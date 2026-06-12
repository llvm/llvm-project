; RUN: llvm-as < %s | llvm-dis > %t0
; RUN: opt -S < %s > %t1
; RUN: diff %t0 %t1
; RUN: verify-uselistorder < %s
; PR6140

; Make sure the flags are serialized/deserialized properly for both
; forward and backward references.

define void @foo() nounwind {
entry:
  br label %first

second:                                           ; preds = %first
  %u = add nuw i32 %a, 0                          ; <i32> [#uses=0]
  %s = add nsw i32 %a, 0                          ; <i32> [#uses=0]
  %us = add nuw nsw i32 %a, 0                     ; <i32> [#uses=0]
  %z = add i32 %a, 0                              ; <i32> [#uses=0]
  %hh = zext nneg i32 %a to i64
  %ll = zext i32 %s to i64
  %ff = uitofp nneg i32 %a to float
  %bb = uitofp i32 %s to float
  %jj = or disjoint i32 %a, 0
  %oo = or i32 %a, 0
  %tu = trunc nuw i32 %a to i16
  %ts = trunc nsw i32 %a to i16
  %tus = trunc nuw nsw i32 %a to i16
  %t = trunc i32 %a to i16
  %tuv = trunc nuw <2 x i32> %aa to <2 x i16>
  %tsv = trunc nsw <2 x i32> %aa to <2 x i16>
  %tusv = trunc nuw nsw <2 x i32> %aa to <2 x i16>
  %tv = trunc <2 x i32> %aa to <2 x i16>
  %ii = icmp samesign ult i32 %a, %z
  %iv = icmp samesign ult <2 x i32> %aa, %aa
  unreachable

first:                                                    ; preds = %entry
  %aa = bitcast <2 x i32> <i32 0, i32 0> to <2 x i32>
  %a = bitcast i32 0 to i32                               ; <i32> [#uses=8]
  %uu = add nuw i32 %a, 0                                 ; <i32> [#uses=0]
  %ss = add nsw i32 %a, 0                                 ; <i32> [#uses=0]
  %uuss = add nuw nsw i32 %a, 0                           ; <i32> [#uses=0]
  %zz = add i32 %a, 0                                     ; <i32> [#uses=0]
  %kk = zext nneg i32 %a to i64
  %rr = zext i32 %ss to i64
  %ww = uitofp nneg i32 %a to float
  %xx = uitofp i32 %ss to float
  %mm = or disjoint i32 %a, 0
  %nn = or i32 %a, 0
  %tuu = trunc nuw i32 %a to i16
  %tss = trunc nsw i32 %a to i16
  %tuss = trunc nuw nsw i32 %a to i16
  %tt = trunc i32 %a to i16
  %ttuv = trunc nuw <2 x i32> %aa to <2 x i16>
  %ttsv = trunc nsw <2 x i32> %aa to <2 x i16>
  %ttusv = trunc nuw nsw <2 x i32> %aa to <2 x i16>
  %ttv = trunc <2 x i32> %aa to <2 x i16>
  %icm = icmp samesign ult i32 %a, %zz
  %icv = icmp samesign ult <2 x i32> %aa, %aa
  br label %second
}
