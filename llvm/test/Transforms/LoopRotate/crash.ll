; RUN: opt -loop-rotate -disable-output -verify-dom-info -verify-loop-info -verify-memoryssa < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; PR8955 - Rotating an outer loop that has a condbr for a latch block.
define void @test1() nounwind ssp {
entry:
  br label %lbl_283

lbl_283:                                          ; preds = %if.end, %entry
  br i1 undef, label %if.else, label %if.then

if.then:                                          ; preds = %lbl_283
  br i1 undef, label %if.end, label %for.condthread-pre-split

for.condthread-pre-split:                         ; preds = %if.then
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %for.condthread-pre-split
  br i1 undef, label %lbl_281, label %for.cond

lbl_281:                                          ; preds = %if.end, %for.cond
  br label %if.end

if.end:                                           ; preds = %lbl_281, %if.then
  br i1 undef, label %lbl_283, label %lbl_281

if.else:                                          ; preds = %lbl_283
  ret void
}

        %struct.relation = type { [4 x i16], i32, [4 x i16], i32, i32 }

define void @test2() {
entry:
        br i1 false, label %bb139, label %bb10.i44
bb10.i44:               ; preds = %entry
        ret void
bb127:          ; preds = %bb139
        br label %bb139
bb139:          ; preds = %bb127, %entry
        br i1 false, label %bb127, label %bb142
bb142:          ; preds = %bb139
        %r91.0.lcssa = phi %struct.relation* [ null, %bb139 ]           ; <%struct.relation*> [#uses=0]
        ret void
}


define void @test3() {
entry:
	br i1 false, label %bb139, label %cond_true
cond_true:		; preds = %entry
	ret void
bb90:		; preds = %bb139
	br i1 false, label %bb136, label %cond_next121
cond_next121:		; preds = %bb90
	br i1 false, label %bb136, label %bb127
bb127:		; preds = %cond_next121
	br label %bb136
bb136:		; preds = %bb127, %cond_next121, %bb90
	%changes.1 = phi i32 [ %changes.2, %bb90 ], [ %changes.2, %cond_next121 ], [ 1, %bb127 ]		; <i32> [#uses=1]
	br label %bb139
bb139:		; preds = %bb136, %entry
	%changes.2 = phi i32 [ %changes.1, %bb136 ], [ 0, %entry ]		; <i32> [#uses=3]
	br i1 false, label %bb90, label %bb142
bb142:		; preds = %bb139
	%changes.2.lcssa = phi i32 [ %changes.2, %bb139 ]		; <i32> [#uses=0]
	ret void
}

define void @test4() {
entry:
	br i1 false, label %cond_false485, label %bb405
bb405:		; preds = %entry
	ret void
cond_false485:		; preds = %entry
	br label %bb830
bb511:		; preds = %bb830
	br i1 false, label %bb816, label %bb830
cond_next667:		; preds = %bb816
	br i1 false, label %cond_next695, label %bb680
bb676:		; preds = %bb680
	br label %bb680
bb680:		; preds = %bb676, %cond_next667
	%iftmp.68.0 = zext i1 false to i8		; <i8> [#uses=1]
	br i1 false, label %bb676, label %cond_next695
cond_next695:		; preds = %bb680, %cond_next667
	%iftmp.68.2 = phi i8 [ %iftmp.68.0, %bb680 ], [ undef, %cond_next667 ]	; <i8> [#uses=0]
	ret void
bb816:		; preds = %bb816, %bb511
	br i1 false, label %cond_next667, label %bb816
bb830:		; preds = %bb511, %cond_false485
	br i1 false, label %bb511, label %bb835
bb835:		; preds = %bb830
	ret void
}

	%struct.NSArray = type { %struct.NSObject }
	%struct.NSObject = type { %struct.objc_class* }
	%struct.NSRange = type { i64, i64 }
	%struct._message_ref_t = type { %struct.NSObject* (%struct.NSObject*, %struct._message_ref_t*, ...)*, %struct.objc_selector* }
	%struct.objc_class = type opaque
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_26" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=1]

define %struct.NSArray* @test5(%struct.NSArray* %self, %struct._message_ref_t* %_cmd) {
entry:
	br label %bb116

bb116:		; preds = %bb131, %entry
	%tmp123 = call %struct.NSRange null( %struct.NSObject* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_26", %struct.NSArray* null )		; <%struct.NSRange> [#uses=1]
	br i1 false, label %bb141, label %bb131

bb131:		; preds = %bb116
	%mrv_gr125 = extractvalue %struct.NSRange %tmp123, 1		; <i64> [#uses=0]
	br label %bb116

bb141:		; preds = %bb116
	ret %struct.NSArray* null
}

define void @test6(i8* %msg) {
entry:
	br label %bb15
bb6:		; preds = %bb15
	%gep.upgrd.1 = zext i32 %offset.1 to i64		; <i64> [#uses=1]
	%tmp11 = getelementptr i8, i8* %msg, i64 %gep.upgrd.1		; <i8*> [#uses=0]
	br label %bb15
bb15:		; preds = %bb6, %entry
	%offset.1 = add i32 0, 1		; <i32> [#uses=2]
	br i1 false, label %bb6, label %bb17
bb17:		; preds = %bb15
	%offset.1.lcssa = phi i32 [ %offset.1, %bb15 ]		; <i32> [#uses=0]
	%payload_type.1.lcssa = phi i32 [ 0, %bb15 ]		; <i32> [#uses=0]
	ret void
}




; PR9523 - Non-canonical loop.
define void @test7(i8* %P) nounwind {
entry:
  indirectbr i8* %P, [label %"3", label %"5"]

"3":                                              ; preds = %"4", %entry
  br i1 undef, label %"5", label %"4"

"4":                                              ; preds = %"3"
  br label %"3"

"5":                                              ; preds = %"3", %entry
  ret void
}

; PR21968
define void @test8(i1 %C, i8* %P) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  br i1 %C, label %l_bad, label %for.body

for.body:                                         ; preds = %for.cond
  indirectbr i8* %P, [label %for.inc, label %l_bad]

for.inc:                                          ; preds = %for.body
  br label %for.cond

l_bad:                                            ; preds = %for.body, %for.cond
  ret void
}
