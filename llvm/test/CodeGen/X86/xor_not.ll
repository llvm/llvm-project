; RUN: llvm-as < %s | llc -march=x86 | grep {not\[lwb\]} | count 4
; RUN: llvm-as < %s | llc -march=x86-64 | grep {not\[lwb\]}  | count 4
define i32 @test(i32 %a, i32 %b) nounwind  {
entry:
        %tmp1not = xor i32 %b, -2
	%tmp3 = and i32 %tmp1not, %a
        %tmp4 = lshr i32 %tmp3, 1
        ret i32 %tmp4
}

define i32 @sum32(i32 %a, i32 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i32 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i32 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i32 %a_addr.0, %b_addr.0
        %tmp4not = xor i32 %tmp3, 2147483647
        %tmp6 = and i32 %tmp4not, %b_addr.0
        %tmp8 = shl i32 %tmp6, 1
        %tmp10 = icmp eq i32 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i32 %tmp3
}

define i16 @sum16(i16 %a, i16 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i16 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i16 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i16 %a_addr.0, %b_addr.0
        %tmp4not = xor i16 %tmp3, 32767
        %tmp6 = and i16 %tmp4not, %b_addr.0
        %tmp8 = shl i16 %tmp6, 1
        %tmp10 = icmp eq i16 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i16 %tmp3
}

define i8 @sum8(i8 %a, i8 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i8 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i8 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i8 %a_addr.0, %b_addr.0
        %tmp4not = xor i8 %tmp3, 127
        %tmp6 = and i8 %tmp4not, %b_addr.0
        %tmp8 = shl i8 %tmp6, 1
        %tmp10 = icmp eq i8 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i8 %tmp3
}

define i32 @test2(i32 %a, i32 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i32 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i32 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i32 %a_addr.0, %b_addr.0
        %tmp4not = xor i32 %tmp3, 2147483646
        %tmp6 = and i32 %tmp4not, %b_addr.0
        %tmp8 = shl i32 %tmp6, 1
        %tmp10 = icmp eq i32 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i32 %tmp3
}

