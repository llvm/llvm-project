; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

@.str1 = external constant [4 x i8]		; <[4 x i8]*> [#uses=1]
@.str2 = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) {
entry:
	br i1 false, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	ret i32 0

cond_next:		; preds = %entry
	%tmp10 = invoke i16 @_ZN12token_stream4openEPKc( i8* null, i8* null ) signext 
			to label %invcont unwind label %cleanup690		; <i16> [#uses=0]

invcont:		; preds = %cond_next
	%tmp15 = invoke i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @.str1, i32 0, i32 0) )
			to label %invcont14 unwind label %cleanup685		; <i32> [#uses=0]

invcont14:		; preds = %invcont
	%tmp17 = invoke i8* @_ZN24lambda_expression_parser10expressionEPP11arglst_node( i8* null, i8** null )
			to label %cond_true22 unwind label %cleanup685

cond_true22:		; preds = %invcont14
	%tmp35 = invoke i32 null( i8* null )
			to label %cond_next56 unwind label %cleanup685		; <i32> [#uses=0]

cond_next56:		; preds = %cond_true22
	%tmp59 = invoke i32 (i8*, ...)* @printf( i8* getelementptr ([5 x i8]* @.str2, i32 0, i32 0) )
			to label %invcont58 unwind label %cleanup685		; <i32> [#uses=0]

invcont58:		; preds = %cond_next56
	invoke void null( i8* null, i8* null, i32 0 )
			to label %invcont72 unwind label %cleanup685

invcont72:		; preds = %invcont58
	%tmp143 = invoke i32 null( i8* null )
			to label %invcont142 unwind label %cleanup685		; <i32> [#uses=0]

invcont142:		; preds = %invcont72
	br i1 false, label %cond_false407, label %cond_true150

cond_true150:		; preds = %invcont142
	ret i32 0

cond_false407:		; preds = %invcont142
	%tmp431 = invoke i8* null( i8* null, i8* null, i32 0, i32* null )
			to label %bb432 unwind label %cleanup685

bb432:		; preds = %bb432, %cond_false407
	%rexp413.7 = phi i8* [ %tmp431, %cond_false407 ], [ %rexp413.7, %bb432 ]
	%tmp434 = icmp eq i8* %rexp413.7, null		; <i1> [#uses=1]
	br i1 %tmp434, label %bb432, label %cond_true437

cond_true437:		; preds = %bb432
	ret i32 0

cleanup685:		; preds = %cond_false407, %invcont72, %invcont58, %cond_next56, %cond_true22, %invcont14, %invcont
	ret i32 0

cleanup690:		; preds = %cond_next
	ret i32 0
}

declare i16 @_ZN12token_stream4openEPKc(i8*, i8*) signext 

declare i32 @printf(i8*, ...)

declare i8* @_ZN24lambda_expression_parser10expressionEPP11arglst_node(i8*, i8**)
