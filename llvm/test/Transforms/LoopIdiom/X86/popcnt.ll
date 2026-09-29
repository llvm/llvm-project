; RUN: opt -passes=loop-idiom < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -S | FileCheck %s

target triple = "x86_64-apple-macosx10.8.0"

;To recognize this pattern:
;int popcount(unsigned long long a) {
;    int c = 0;
;    while (a) {
;        c++;
;        a &= a - 1;
;    }
;    return c;
;}
;
; CHECK: entry
; CHECK: llvm.ctpop.i64
; CHECK: ret
define i32 @popcount(i64 %a) nounwind uwtable readnone ssp {
entry:
  %tobool3 = icmp eq i64 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i64 %a.addr.04, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}

; To recognize this pattern:
;int popcount(unsigned long long a, int mydata1, int mydata2) {
;    int c = 0;
;    while (a) {
;        c++;
;        a &= a - 1;
;        mydata1 *= c;
;        mydata2 *= (int)a;
;    }
;    return c + mydata1 + mydata2;
;}
; CHECK: entry
; CHECK: llvm.ctpop.i64
; CHECK: ret
define i32 @popcount2(i64 %a, i32 %mydata1, i32 %mydata2) nounwind uwtable readnone ssp {
entry:
  %tobool9 = icmp eq i64 %a, 0
  br i1 %tobool9, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.013 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %mydata2.addr.012 = phi i32 [ %mul1, %while.body ], [ %mydata2, %entry ]
  %mydata1.addr.011 = phi i32 [ %mul, %while.body ], [ %mydata1, %entry ]
  %a.addr.010 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.013, 1
  %sub = add i64 %a.addr.010, -1
  %and = and i64 %sub, %a.addr.010
  %mul = mul nsw i32 %inc, %mydata1.addr.011
  %conv = trunc i64 %and to i32
  %mul1 = mul nsw i32 %conv, %mydata2.addr.012
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %mydata2.addr.0.lcssa = phi i32 [ %mydata2, %entry ], [ %mul1, %while.body ]
  %mydata1.addr.0.lcssa = phi i32 [ %mydata1, %entry ], [ %mul, %while.body ]
  %add = add i32 %mydata2.addr.0.lcssa, %mydata1.addr.0.lcssa
  %add2 = add i32 %add, %c.0.lcssa
  ret i32 %add2
}

; Some variants once cause crash

define i32 @PopCntCrash1(i64 %a) nounwind uwtable readnone ssp {
entry:
  %tobool3 = icmp eq i64 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %t = add i32 %c.05, %c.05
  %inc = add nsw i32 %t, 1
  %sub = add i64 %a.addr.04, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa

; CHECK: entry
; CHECK: ret
}

define i32 @PopCntCrash2(i64 %a, i32 %b) nounwind uwtable readnone ssp {
entry:
  %tobool3 = icmp eq i64 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ %b, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i64 %a.addr.04, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}

define i32 @PopCntCrash3(i64 %a, i32 %x) {
entry:
  %tobool3 = icmp eq i64 %a, 0
  %cmp = icmp eq i32 %x, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i64 %a.addr.04, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}

; The a & (a - 1) in the loop is a & (b - 1) in this code. Make sure we don't
; convert it.
define i32 @popcount_bad(i64 %a, i64 %b) nounwind uwtable readnone ssp {
; CHECK-LABEL: @popcount_bad(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TOBOOL3:%.*]] = icmp eq i64 [[A:%.*]], 0
; CHECK-NEXT:    br i1 [[TOBOOL3]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CHECK:       while.body.preheader:
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.body:
; CHECK-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; CHECK-NEXT:    [[A_ADDR_04:%.*]] = phi i64 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; CHECK-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; CHECK-NEXT:    [[SUB:%.*]] = add i64 [[B:%.*]], -1
; CHECK-NEXT:    [[AND]] = and i64 [[SUB]], [[A_ADDR_04]]
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i64 [[AND]], 0
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; CHECK:       while.end.loopexit:
; CHECK-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; CHECK-NEXT:    br label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CHECK-NEXT:    ret i32 [[C_0_LCSSA]]
;
entry:
  %tobool3 = icmp eq i64 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i64 %b, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}
