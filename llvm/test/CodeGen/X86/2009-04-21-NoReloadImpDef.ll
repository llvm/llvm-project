; RUN: llc -mtriple=i386-apple-darwin10.0 -relocation-model=pic -asm-verbose=false \
; RUN:     -mcpu=generic -disable-fp-elim -mattr=-sse41,-sse3,+sse2 -post-RA-scheduler=false -regalloc=basic < %s | \
; RUN:   FileCheck %s
; rdar://6808032

; CHECK: pextrw $14
; CHECK-NEXT: shrl $8
; CHECK-NEXT: pinsrw

define void @update(i8** %args_list) nounwind {
entry:
	%cmp.i = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %cmp.i, label %if.then.i, label %test_cl.exit

if.then.i:		; preds = %entry
	%val = load <16 x i8> addrspace(1)* null		; <<16 x i8>> [#uses=8]
	%tmp10.i = shufflevector <16 x i8> <i8 0, i8 0, i8 0, i8 undef, i8 0, i8 undef, i8 0, i8 undef, i8 undef, i8 undef, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef>, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 4, i32 undef, i32 6, i32 undef, i32 29, i32 undef, i32 10, i32 11, i32 12, i32 undef, i32 undef, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp17.i = shufflevector <16 x i8> %tmp10.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 18, i32 4, i32 undef, i32 6, i32 undef, i32 8, i32 undef, i32 10, i32 11, i32 12, i32 undef, i32 undef, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp24.i = shufflevector <16 x i8> %tmp17.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 24, i32 6, i32 undef, i32 8, i32 undef, i32 10, i32 11, i32 12, i32 undef, i32 undef, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp31.i = shufflevector <16 x i8> %tmp24.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 undef, i32 8, i32 undef, i32 10, i32 11, i32 12, i32 21, i32 undef, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp38.i = shufflevector <16 x i8> %tmp31.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 27, i32 8, i32 undef, i32 10, i32 11, i32 12, i32 13, i32 undef, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp45.i = shufflevector <16 x i8> %tmp38.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 undef, i32 10, i32 11, i32 12, i32 13, i32 29, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp52.i = shufflevector <16 x i8> %tmp45.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 21, i32 10, i32 11, i32 12, i32 13, i32 14, i32 undef>		; <<16 x i8>> [#uses=1]
	%tmp59.i = shufflevector <16 x i8> %tmp52.i, <16 x i8> %val, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 20>		; <<16 x i8>> [#uses=1]
	store <16 x i8> %tmp59.i, <16 x i8> addrspace(1)* null
	ret void

test_cl.exit:		; preds = %entry
	ret void
}
