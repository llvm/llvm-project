; RUN: llvm-as < %s | opt -simplifycfg  | llvm-dis | grep select
; RUN: llvm-as < %s | opt -simplifycfg  | llvm-dis | grep br | count 2

define i32 @t2(i32 %a, i32 %b, i32 %c) nounwind  {
entry:
        %tmp1 = icmp eq i32 %b, 0
        br i1 %tmp1, label %bb1, label %bb3

bb1:            ; preds = %entry
	%tmp2 = icmp sgt i32 %c, 1
	br i1 %tmp2, label %bb2, label %bb3

bb2:		; preds = bb1
	%tmp3 = add i32 %a, 1
	br label %bb3

bb3:		; preds = %bb2, %entry
	%tmp4 = phi i32 [ %b, %entry ], [ %a, %bb1 ], [ %tmp3, %bb2 ]
        %tmp5 = sub i32 %tmp4, 1
	ret i32 %tmp5
}
