; RUN: opt < %s -loop-reduce -disable-output
; PR 2537

define void @a() {
entry:
        br label %dobody

dobody:         ; preds = %dobody, %entry
        %y.0 = phi i128 [ 0, %entry ], [ %add, %dobody ]
        %x.0 = phi i128 [ 0, %entry ], [ %add2, %dobody ]
        %add = add i128 %y.0, u0x10000000000000000
        %add2 = add i128 %x.0, u0x1000000000000
        call void @b( i128 %add )
        %cmp = icmp ult i128 %add2, u0x10000000000000000
        br i1 %cmp, label %dobody, label %afterdo

afterdo:                ; preds = %dobody
        ret void
}

declare void @b(i128 %add)
