; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -asm-verbose=false -post-RA-scheduler=true | FileCheck %s

declare void @bar(i32)
declare void @car(i32)
declare void @dar(i32)
declare void @ear(i32)
declare void @far(i32)
declare i1 @qux()

@GHJK = global i32 0
@HABC = global i32 0

; BranchFolding should tail-merge the stores since they all precede
; direct branches to the same place.

; CHECK: tail_merge_me:
; CHECK-NOT:  GHJK
; CHECK:      movl $0, GHJK(%rip)
; CHECK-NEXT: movl $1, HABC(%rip)
; CHECK-NOT:  GHJK

define void @tail_merge_me() nounwind {
entry:
  %a = call i1 @qux()
  br i1 %a, label %A, label %next
next:
  %b = call i1 @qux()
  br i1 %b, label %B, label %C

A:
  call void @bar(i32 0)
  store i32 0, i32* @GHJK
  br label %M

B:
  call void @car(i32 1)
  store i32 0, i32* @GHJK
  br label %M

C:
  call void @dar(i32 2)
  store i32 0, i32* @GHJK
  br label %M

M:
  store i32 1, i32* @HABC
  %c = call i1 @qux()
  br i1 %c, label %return, label %altret

return:
  call void @ear(i32 1000)
  ret void
altret:
  call void @far(i32 1001)
  ret void
}

declare i8* @choose(i8*, i8*)

; BranchFolding should tail-duplicate the indirect jump to avoid
; redundant branching.

; CHECK: tail_duplicate_me:
; CHECK:      movl $0, GHJK(%rip)
; CHECK-NEXT: jmpq *%r
; CHECK:      movl $0, GHJK(%rip)
; CHECK-NEXT: jmpq *%r
; CHECK:      movl $0, GHJK(%rip)
; CHECK-NEXT: jmpq *%r

define void @tail_duplicate_me() nounwind {
entry:
  %a = call i1 @qux()
  %c = call i8* @choose(i8* blockaddress(@tail_duplicate_me, %return),
                        i8* blockaddress(@tail_duplicate_me, %altret))
  br i1 %a, label %A, label %next
next:
  %b = call i1 @qux()
  br i1 %b, label %B, label %C

A:
  call void @bar(i32 0)
  store i32 0, i32* @GHJK
  br label %M

B:
  call void @car(i32 1)
  store i32 0, i32* @GHJK
  br label %M

C:
  call void @dar(i32 2)
  store i32 0, i32* @GHJK
  br label %M

M:
  indirectbr i8* %c, [label %return, label %altret]

return:
  call void @ear(i32 1000)
  ret void
altret:
  call void @far(i32 1001)
  ret void
}

; BranchFolding shouldn't try to merge the tails of two blocks
; with only a branch in common, regardless of the fallthrough situation.

; CHECK: dont_merge_oddly:
; CHECK-NOT:   ret
; CHECK:        ucomiss %xmm{{[0-2]}}, %xmm{{[0-2]}}
; CHECK-NEXT:   jbe .LBB2_3
; CHECK-NEXT:   ucomiss %xmm{{[0-2]}}, %xmm{{[0-2]}}
; CHECK-NEXT:   ja .LBB2_4
; CHECK-NEXT:   jmp .LBB2_2
; CHECK-NEXT: .LBB2_3:
; CHECK-NEXT:   ucomiss %xmm{{[0-2]}}, %xmm{{[0-2]}}
; CHECK-NEXT:   jbe .LBB2_2
; CHECK-NEXT: .LBB2_4:
; CHECK-NEXT:   xorb %al, %al
; CHECK-NEXT:   ret
; CHECK-NEXT: .LBB2_2:
; CHECK-NEXT:   movb $1, %al
; CHECK-NEXT:   ret

define i1 @dont_merge_oddly(float* %result) nounwind {
entry:
  %tmp4 = getelementptr float* %result, i32 2
  %tmp5 = load float* %tmp4, align 4
  %tmp7 = getelementptr float* %result, i32 4
  %tmp8 = load float* %tmp7, align 4
  %tmp10 = getelementptr float* %result, i32 6
  %tmp11 = load float* %tmp10, align 4
  %tmp12 = fcmp olt float %tmp8, %tmp11
  br i1 %tmp12, label %bb, label %bb21

bb:
  %tmp23469 = fcmp olt float %tmp5, %tmp8
  br i1 %tmp23469, label %bb26, label %bb30

bb21:
  %tmp23 = fcmp olt float %tmp5, %tmp11
  br i1 %tmp23, label %bb26, label %bb30

bb26:
  ret i1 0

bb30:
  ret i1 1
}

; Do any-size tail-merging when two candidate blocks will both require
; an unconditional jump to complete a two-way conditional branch.

; CHECK: c_expand_expr_stmt:
;
; This test only works when register allocation happens to use %rax for both
; load addresses.
;
; CHE:        jmp .LBB3_11
; CHE-NEXT: .LBB3_9:
; CHE-NEXT:   movq 8(%rax), %rax
; CHE-NEXT:   xorb %dl, %dl
; CHE-NEXT:   movb 16(%rax), %al
; CHE-NEXT:   cmpb $16, %al
; CHE-NEXT:   je .LBB3_11
; CHE-NEXT:   cmpb $23, %al
; CHE-NEXT:   jne .LBB3_14
; CHE-NEXT: .LBB3_11:

%0 = type { %struct.rtx_def* }
%struct.lang_decl = type opaque
%struct.rtx_def = type { i16, i8, i8, [1 x %union.rtunion] }
%struct.tree_decl = type { [24 x i8], i8*, i32, %union.tree_node*, i32, i8, i8, i8, i8, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %struct.rtx_def*, %union..2anon, %0, %union.tree_node*, %struct.lang_decl* }
%union..2anon = type { i32 }
%union.rtunion = type { i8* }
%union.tree_node = type { %struct.tree_decl }

define fastcc void @c_expand_expr_stmt(%union.tree_node* %expr) nounwind {
entry:
  %tmp4 = load i8* null, align 8                  ; <i8> [#uses=3]
  switch i8 %tmp4, label %bb3 [
    i8 18, label %bb
  ]

bb:                                               ; preds = %entry
  switch i32 undef, label %bb1 [
    i32 0, label %bb2.i
    i32 37, label %bb.i
  ]

bb.i:                                             ; preds = %bb
  switch i32 undef, label %bb1 [
    i32 0, label %lvalue_p.exit
  ]

bb2.i:                                            ; preds = %bb
  br label %bb3

lvalue_p.exit:                                    ; preds = %bb.i
  %tmp21 = load %union.tree_node** null, align 8  ; <%union.tree_node*> [#uses=3]
  %tmp22 = getelementptr inbounds %union.tree_node* %tmp21, i64 0, i32 0, i32 0, i64 0 ; <i8*> [#uses=1]
  %tmp23 = load i8* %tmp22, align 8               ; <i8> [#uses=1]
  %tmp24 = zext i8 %tmp23 to i32                  ; <i32> [#uses=1]
  switch i32 %tmp24, label %lvalue_p.exit4 [
    i32 0, label %bb2.i3
    i32 2, label %bb.i1
  ]

bb.i1:                                            ; preds = %lvalue_p.exit
  %tmp25 = getelementptr inbounds %union.tree_node* %tmp21, i64 0, i32 0, i32 2 ; <i32*> [#uses=1]
  %tmp26 = bitcast i32* %tmp25 to %union.tree_node** ; <%union.tree_node**> [#uses=1]
  %tmp27 = load %union.tree_node** %tmp26, align 8 ; <%union.tree_node*> [#uses=2]
  %tmp28 = getelementptr inbounds %union.tree_node* %tmp27, i64 0, i32 0, i32 0, i64 16 ; <i8*> [#uses=1]
  %tmp29 = load i8* %tmp28, align 8               ; <i8> [#uses=1]
  %tmp30 = zext i8 %tmp29 to i32                  ; <i32> [#uses=1]
  switch i32 %tmp30, label %lvalue_p.exit4 [
    i32 0, label %bb2.i.i2
    i32 2, label %bb.i.i
  ]

bb.i.i:                                           ; preds = %bb.i1
  %tmp34 = tail call fastcc i32 @lvalue_p(%union.tree_node* null) nounwind ; <i32> [#uses=1]
  %phitmp = icmp ne i32 %tmp34, 0                 ; <i1> [#uses=1]
  br label %lvalue_p.exit4

bb2.i.i2:                                         ; preds = %bb.i1
  %tmp35 = getelementptr inbounds %union.tree_node* %tmp27, i64 0, i32 0, i32 0, i64 8 ; <i8*> [#uses=1]
  %tmp36 = bitcast i8* %tmp35 to %union.tree_node** ; <%union.tree_node**> [#uses=1]
  %tmp37 = load %union.tree_node** %tmp36, align 8 ; <%union.tree_node*> [#uses=1]
  %tmp38 = getelementptr inbounds %union.tree_node* %tmp37, i64 0, i32 0, i32 0, i64 16 ; <i8*> [#uses=1]
  %tmp39 = load i8* %tmp38, align 8               ; <i8> [#uses=1]
  switch i8 %tmp39, label %bb2 [
    i8 16, label %lvalue_p.exit4
    i8 23, label %lvalue_p.exit4
  ]

bb2.i3:                                           ; preds = %lvalue_p.exit
  %tmp40 = getelementptr inbounds %union.tree_node* %tmp21, i64 0, i32 0, i32 0, i64 8 ; <i8*> [#uses=1]
  %tmp41 = bitcast i8* %tmp40 to %union.tree_node** ; <%union.tree_node**> [#uses=1]
  %tmp42 = load %union.tree_node** %tmp41, align 8 ; <%union.tree_node*> [#uses=1]
  %tmp43 = getelementptr inbounds %union.tree_node* %tmp42, i64 0, i32 0, i32 0, i64 16 ; <i8*> [#uses=1]
  %tmp44 = load i8* %tmp43, align 8               ; <i8> [#uses=1]
  switch i8 %tmp44, label %bb2 [
    i8 16, label %lvalue_p.exit4
    i8 23, label %lvalue_p.exit4
  ]

lvalue_p.exit4:                                   ; preds = %bb2.i3, %bb2.i3, %bb2.i.i2, %bb2.i.i2, %bb.i.i, %bb.i1, %lvalue_p.exit
  %tmp45 = phi i1 [ %phitmp, %bb.i.i ], [ false, %bb2.i.i2 ], [ false, %bb2.i.i2 ], [ false, %bb.i1 ], [ false, %bb2.i3 ], [ false, %bb2.i3 ], [ false, %lvalue_p.exit ] ; <i1> [#uses=1]
  %tmp46 = icmp eq i8 %tmp4, 0                    ; <i1> [#uses=1]
  %or.cond = or i1 %tmp45, %tmp46                 ; <i1> [#uses=1]
  br i1 %or.cond, label %bb2, label %bb3

bb1:                                              ; preds = %bb2.i.i, %bb.i, %bb
  %.old = icmp eq i8 %tmp4, 23                    ; <i1> [#uses=1]
  br i1 %.old, label %bb2, label %bb3

bb2:                                              ; preds = %bb1, %lvalue_p.exit4, %bb2.i3, %bb2.i.i2
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1, %lvalue_p.exit4, %bb2.i, %entry
  %expr_addr.0 = phi %union.tree_node* [ null, %bb2 ], [ %expr, %bb2.i ], [ %expr, %entry ], [ %expr, %bb1 ], [ %expr, %lvalue_p.exit4 ] ; <%union.tree_node*> [#uses=0]
  unreachable
}

declare fastcc i32 @lvalue_p(%union.tree_node* nocapture) nounwind readonly

declare fastcc %union.tree_node* @default_conversion(%union.tree_node*) nounwind


; If one tail merging candidate falls through into the other,
; tail merging is likely profitable regardless of how few
; instructions are involved. This function should have only
; one ret instruction.

; CHECK: foo:
; CHECK:        callq func
; CHECK-NEXT: .LBB4_2:
; CHECK-NEXT:   popq
; CHECK-NEXT:   ret

define void @foo(i1* %V) nounwind {
entry:
  %t0 = icmp eq i1* %V, null
  br i1 %t0, label %return, label %bb

bb:
  call void @func()
  ret void

return:
  ret void
}

declare void @func()

; one - One instruction may be tail-duplicated even with optsize.

; CHECK: one:
; CHECK: movl $0, XYZ(%rip)
; CHECK: movl $0, XYZ(%rip)

@XYZ = external global i32

define void @one() nounwind optsize {
entry:
  %0 = icmp eq i32 undef, 0
  br i1 %0, label %bbx, label %bby

bby:
  switch i32 undef, label %bb7 [
    i32 16, label %return
  ]

bb7:
  store volatile i32 0, i32* @XYZ
  unreachable

bbx:
  switch i32 undef, label %bb12 [
    i32 128, label %return
  ]

bb12:
  store volatile i32 0, i32* @XYZ
  unreachable

return:
  ret void
}

; two - Same as one, but with two instructions in the common
; tail instead of one. This is too much to be merged, given
; the optsize attribute.

; CHECK: two:
; CHECK-NOT: XYZ
; CHECK: ret
; CHECK: movl $0, XYZ(%rip)
; CHECK: movl $1, XYZ(%rip)
; CHECK-NOT: XYZ

define void @two() nounwind optsize {
entry:
  %0 = icmp eq i32 undef, 0
  br i1 %0, label %bbx, label %bby

bby:
  switch i32 undef, label %bb7 [
    i32 16, label %return
  ]

bb7:
  store volatile i32 0, i32* @XYZ
  store volatile i32 1, i32* @XYZ
  unreachable

bbx:
  switch i32 undef, label %bb12 [
    i32 128, label %return
  ]

bb12:
  store volatile i32 0, i32* @XYZ
  store volatile i32 1, i32* @XYZ
  unreachable

return:
  ret void
}

; two_nosize - Same as two, but without the optsize attribute.
; Now two instructions are enough to be tail-duplicated.

; CHECK: two_nosize:
; CHECK: movl $0, XYZ(%rip)
; CHECK: movl $1, XYZ(%rip)
; CHECK: movl $0, XYZ(%rip)
; CHECK: movl $1, XYZ(%rip)

define void @two_nosize() nounwind {
entry:
  %0 = icmp eq i32 undef, 0
  br i1 %0, label %bbx, label %bby

bby:
  switch i32 undef, label %bb7 [
    i32 16, label %return
  ]

bb7:
  store volatile i32 0, i32* @XYZ
  store volatile i32 1, i32* @XYZ
  unreachable

bbx:
  switch i32 undef, label %bb12 [
    i32 128, label %return
  ]

bb12:
  store volatile i32 0, i32* @XYZ
  store volatile i32 1, i32* @XYZ
  unreachable

return:
  ret void
}

; Tail-merging should merge the two ret instructions since one side
; can fall-through into the ret and the other side has to branch anyway.

; CHECK: TESTE:
; CHECK: ret
; CHECK-NOT: ret
; CHECK: size TESTE

define i64 @TESTE(i64 %parami, i64 %paraml) nounwind readnone {
entry:
  %cmp = icmp slt i64 %parami, 1                  ; <i1> [#uses=1]
  %varx.0 = select i1 %cmp, i64 1, i64 %parami    ; <i64> [#uses=1]
  %cmp410 = icmp slt i64 %paraml, 1               ; <i1> [#uses=1]
  br i1 %cmp410, label %for.end, label %bb.nph

bb.nph:                                           ; preds = %entry
  %tmp15 = mul i64 %paraml, %parami                   ; <i64> [#uses=1]
  ret i64 %tmp15

for.end:                                          ; preds = %entry
  ret i64 %varx.0
}
