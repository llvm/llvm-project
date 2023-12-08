; RUN: llc < %s -mtriple=x86_64--
; PR3886

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind {
entry:
        %a = call <1 x i64> @bar()
        %tmp5.i = extractelement <1 x i64> %a, i32 0
        %tmp11 = bitcast i64 %tmp5.i to <1 x i64>
        %tmp8 = extractelement <1 x i64> %tmp11, i32 0
        %call6 = call i32 (i64) @foo(i64 %tmp8)
        ret i32 undef
}

declare i32 @foo(i64)

declare <1 x i64> @bar()
