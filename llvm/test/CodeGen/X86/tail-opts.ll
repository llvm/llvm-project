declare dso_local void @bar(i32)
declare dso_local void @car(i32)
declare dso_local void @dar(i32)
declare dso_local void @ear(i32)
declare dso_local void @far(i32)
declare i1 @qux()
%0 = type { %struct.rtx_def* }
%struct.lang_decl = type opaque
%struct.rtx_def = type { i16, i8, i8, [1 x %union.rtunion] }
%struct.tree_decl = type { [24 x i8], i8*, i32, %union.tree_node*, i32, i8, i8, i8, i8, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %struct.rtx_def*, %union..2anon, %0, %union.tree_node*, %struct.lang_decl* }
%union..2anon = type { i32 }
%union.rtunion = type { i8* }
%union.tree_node = type { %struct.tree_decl }

define fastcc void @c_expand_expr_stmt(%union.tree_node* %expr) nounwind {
entry:
  %tmp4 = load i8, i8* null, align 8                  ; <i8> [#uses=3]
  br label %bb3

lvalue_p.exit:                                    ; preds = %bb.i
  %tmp21 = load %union.tree_node*, %union.tree_node** null, align 8  ; <%union.tree_node*> [#uses=3]
  %tmp22 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp21, i64 0, i32 0, i32 0, i64 0 ; <i8*> [#uses=1]
  %tmp23 = load i8, i8* %tmp22, align 8               ; <i8> [#uses=1]
  %tmp24 = zext i8 %tmp23 to i32                  ; <i32> [#uses=1]
  switch i32 %tmp24, label %lvalue_p.exit4 [
    i32 0, label %bb2.i3
    i32 2, label %bb.i1
  ]

bb.i1:                                            ; preds = %lvalue_p.exit
  %tmp25 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp21, i64 0, i32 0, i32 2 ; <i32*> [#uses=1]
  %tmp26 = bitcast i32* %tmp25 to %union.tree_node** ; <%union.tree_node**> [#uses=1]
  %tmp27 = load %union.tree_node*, %union.tree_node** %tmp26, align 8 ; <%union.tree_node*> [#uses=2]
  %tmp28 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp27, i64 0, i32 0, i32 0, i64 16 ; <i8*> [#uses=1]
  %tmp29 = load i8, i8* %tmp28, align 8               ; <i8> [#uses=1]
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
  %tmp35 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp27, i64 0, i32 0, i32 0, i64 8 ; <i8*> [#uses=1]
  %tmp36 = bitcast i8* %tmp35 to %union.tree_node** ; <%union.tree_node**> [#uses=1]
  %tmp37 = load %union.tree_node*, %union.tree_node** %tmp36, align 8 ; <%union.tree_node*> [#uses=1]
  %tmp38 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp37, i64 0, i32 0, i32 0, i64 16 ; <i8*> [#uses=1]
  %tmp39 = load i8, i8* %tmp38, align 8               ; <i8> [#uses=1]
  switch i8 %tmp39, label %bb2 [
    i8 16, label %lvalue_p.exit4
    i8 23, label %lvalue_p.exit4
  ]

bb2.i3:                                           ; preds = %lvalue_p.exit
  %tmp40 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp21, i64 0, i32 0, i32 0, i64 8 ; <i8*> [#uses=1]
  %tmp41 = bitcast i8* %tmp40 to %union.tree_node** ; <%union.tree_node**> [#uses=1]
  %tmp42 = load %union.tree_node*, %union.tree_node** %tmp41, align 8 ; <%union.tree_node*> [#uses=1]
  %tmp43 = getelementptr inbounds %union.tree_node, %union.tree_node* %tmp42, i64 0, i32 0, i32 0, i64 16 ; <i8*> [#uses=1]
  %tmp44 = load i8, i8* %tmp43, align 8               ; <i8> [#uses=1]
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
