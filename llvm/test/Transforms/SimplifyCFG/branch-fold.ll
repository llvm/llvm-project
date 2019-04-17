; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @test(i32* %P, i32* %Q, i1 %A, i1 %B) {
; CHECK: test
; CHECK: br i1
; CHECK-NOT: br i1
; CHECK: ret
; CHECK: ret

entry:
        br i1 %A, label %a, label %b
a:
        br i1 %B, label %b, label %c
b:
        store i32 123, i32* %P
        ret void
c:
        ret void
}

; rdar://10554090
define zeroext i1 @test2(i64 %i0, i64 %i1) nounwind uwtable readonly ssp {
entry:
; CHECK: test2
; CHECK: br i1
  %and.i.i = and i64 %i0, 281474976710655
  %and.i11.i = and i64 %i1, 281474976710655
  %or.cond = icmp eq i64 %and.i.i, %and.i11.i
  br i1 %or.cond, label %c, label %a

a:
; CHECK: br
  %shr.i4.i = lshr i64 %i0, 48
  %and.i5.i = and i64 %shr.i4.i, 32767
  %shr.i.i = lshr i64 %i1, 48
  %and.i2.i = and i64 %shr.i.i, 32767
  %cmp9.i = icmp ult i64 %and.i5.i, %and.i2.i
  br i1 %cmp9.i, label %c, label %b

b:
; CHECK-NOT: br
  %shr.i13.i9 = lshr i64 %i1, 48
  %and.i14.i10 = and i64 %shr.i13.i9, 32767
  %shr.i.i11 = lshr i64 %i0, 48
  %and.i11.i12 = and i64 %shr.i.i11, 32767
  %phitmp = icmp uge i64 %and.i14.i10, %and.i11.i12
  br label %c

c:
  %o2 = phi i1 [ false, %a ], [ %phitmp, %b ], [ false, %entry ]
  ret i1 %o2
}

; PR13180
define void @pr13180(i8 %p) {
entry:
  %tobool = icmp eq i8 %p, 0
  br i1 %tobool, label %cond.false, label %cond.true

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  %phitmp = icmp eq i8 %p, 0
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i1 [ undef, %cond.true ], [ %phitmp, %cond.false ]
  unreachable
}
