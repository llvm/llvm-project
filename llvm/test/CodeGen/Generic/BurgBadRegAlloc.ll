; RUN: llvm-as < %s | llc

;; Register allocation is doing a very poor job on this routine from yyparse
;; in Burg:
;; -- at least two long-lived values are being allocated to %o? registers
;; -- even worse, those registers are being saved and restored repeatedly
;;    at function calls, even though there are no intervening uses.
;; -- outgoing args of some function calls have to be swapped, causing
;;    another write/read from stack to do the exchange (use -dregalloc=y).
;;	
%Arity = type %struct.arity*
	%Binding = type %struct.binding*
	%DeltaCost = type [4 x i16]
	%Dimension = type %struct.dimension*
	%Index_Map = type { i32, %Item_Set* }
	%IntList = type %struct.intlist*
	%Item = type { %DeltaCost, %Rule }
	%ItemArray = type %Item*
	%Item_Set = type %struct.item_set*
	%List = type %struct.list*
	%Mapping = type %struct.mapping*
	%NonTerminal = type %struct.nonterminal*
	%Operator = type %struct.operator*
	%Pattern = type %struct.pattern*
	%PatternAST = type %struct.patternAST*
	%Plank = type %struct.plank*
	%PlankMap = type %struct.plankMap*
	%ReadFn = type i32 ()*
	%Rule = type %struct.rule*
	%RuleAST = type %struct.ruleAST*
	%StateMap = type %struct.stateMap*
	%StrTableElement = type %struct.strTableElement*
	%Symbol = type %struct.symbol*
	%Table = type %struct.table*
	%YYSTYPE = type { %IntList }
	%struct.arity = type { i32, %List }
	%struct.binding = type { i8*, i32 }
	%struct.dimension = type { i16*, %Index_Map, %Mapping, i32, %PlankMap }
	%struct.index_map = type { i32, %Item_Set* }
	%struct.intlist = type { i32, %IntList }
	%struct.item = type { %DeltaCost, %Rule }
	%struct.item_set = type { i32, i32, %Operator, [2 x %Item_Set], %Item_Set, i16*, %ItemArray, %ItemArray }
	%struct.list = type { i8*, %List }
	%struct.mapping = type { %List*, i32, i32, i32, %Item_Set* }
	%struct.nonterminal = type { i8*, i32, i32, i32, %PlankMap, %Rule }
	%struct.operator = type { i8*, i32, i32, i32, i32, i32, %Table }
	%struct.pattern = type { %NonTerminal, %Operator, [2 x %NonTerminal] }
	%struct.patternAST = type { %Symbol, i8*, %List }
	%struct.plank = type { i8*, %List, i32 }
	%struct.plankMap = type { %List, i32, %StateMap }
	%struct.rule = type { %DeltaCost, i32, i32, i32, %NonTerminal, %Pattern, i32 }
	%struct.ruleAST = type { i8*, %PatternAST, i32, %IntList, %Rule, %StrTableElement, %StrTableElement }
	%struct.stateMap = type { i8*, %Plank, i32, i16* }
	%struct.strTableElement = type { i8*, %IntList, i8* }
	%struct.symbol = type { i8*, i32, { %Operator } }
	%struct.table = type { %Operator, %List, i16*, [2 x %Dimension], %Item_Set* }
@yylval = external global %YYSTYPE		; <%YYSTYPE*> [#uses=1]
@yylhs = external global [25 x i16]		; <[25 x i16]*> [#uses=1]
@yylen = external global [25 x i16]		; <[25 x i16]*> [#uses=1]
@yydefred = external global [43 x i16]		; <[43 x i16]*> [#uses=1]
@yydgoto = external global [12 x i16]		; <[12 x i16]*> [#uses=1]
@yysindex = external global [43 x i16]		; <[43 x i16]*> [#uses=2]
@yyrindex = external global [43 x i16]		; <[43 x i16]*> [#uses=1]
@yygindex = external global [12 x i16]		; <[12 x i16]*> [#uses=1]
@yytable = external global [263 x i16]		; <[263 x i16]*> [#uses=4]
@yycheck = external global [263 x i16]		; <[263 x i16]*> [#uses=4]
@yynerrs = external global i32		; <i32*> [#uses=3]
@yyerrflag = external global i32		; <i32*> [#uses=6]
@yychar = external global i32		; <i32*> [#uses=15]
@yyssp = external global i16*		; <i16**> [#uses=15]
@yyvsp = external global %YYSTYPE*		; <%YYSTYPE**> [#uses=30]
@yyval = external global %YYSTYPE		; <%YYSTYPE*> [#uses=1]
@yyss = external global i16*		; <i16**> [#uses=3]
@yysslim = external global i16*		; <i16**> [#uses=3]
@yyvs = external global %YYSTYPE*		; <%YYSTYPE**> [#uses=1]
@.LC01 = external global [13 x i8]		; <[13 x i8]*> [#uses=1]
@.LC1 = external global [20 x i8]		; <[20 x i8]*> [#uses=1]

define i32 @yyparse() {
bb0:
	store i32 0, i32* @yynerrs
	store i32 0, i32* @yyerrflag
	store i32 -1, i32* @yychar
	%reg113 = load i16** @yyss		; <i16*> [#uses=1]
	%cond581 = icmp ne i16* %reg113, null		; <i1> [#uses=1]
	br i1 %cond581, label %bb3, label %bb2

bb2:		; preds = %bb0
	%reg584 = call i32 @yygrowstack( )		; <i32> [#uses=1]
	%cond584 = icmp ne i32 %reg584, 0		; <i1> [#uses=1]
	br i1 %cond584, label %bb113, label %bb3

bb3:		; preds = %bb2, %bb0
	%reg115 = load i16** @yyss		; <i16*> [#uses=1]
	store i16* %reg115, i16** @yyssp
	%reg116 = load %YYSTYPE** @yyvs		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg116, %YYSTYPE** @yyvsp
	%reg117 = load i16** @yyssp		; <i16*> [#uses=1]
	store i16 0, i16* %reg117
	br label %bb4

bb4:		; preds = %bb112, %bb102, %bb35, %bb31, %bb15, %bb14, %bb3
	%reg458 = phi i32 [ %reg476, %bb112 ], [ 1, %bb102 ], [ %reg458, %bb35 ], [ %cast768, %bb31 ], [ %cast658, %bb15 ], [ %cast658, %bb14 ], [ 0, %bb3 ]		; <i32> [#uses=2]
	%reg458-idxcast = zext i32 %reg458 to i64		; <i64> [#uses=3]
	%reg594 = getelementptr [43 x i16]* @yydefred, i64 0, i64 %reg458-idxcast		; <i16*> [#uses=1]
	%reg125 = load i16* %reg594		; <i16> [#uses=1]
	%cast599 = sext i16 %reg125 to i32		; <i32> [#uses=2]
	%cond600 = icmp ne i32 %cast599, 0		; <i1> [#uses=1]
	br i1 %cond600, label %bb36, label %bb5

bb5:		; preds = %bb4
	%reg127 = load i32* @yychar		; <i32> [#uses=1]
	%cond603 = icmp sge i32 %reg127, 0		; <i1> [#uses=1]
	br i1 %cond603, label %bb8, label %bb6

bb6:		; preds = %bb5
	%reg607 = call i32 @yylex( )		; <i32> [#uses=1]
	store i32 %reg607, i32* @yychar
	%reg129 = load i32* @yychar		; <i32> [#uses=1]
	%cond609 = icmp sge i32 %reg129, 0		; <i1> [#uses=1]
	br i1 %cond609, label %bb8, label %bb7

bb7:		; preds = %bb6
	store i32 0, i32* @yychar
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb5
	%reg615 = getelementptr [43 x i16]* @yysindex, i64 0, i64 %reg458-idxcast		; <i16*> [#uses=1]
	%reg137 = load i16* %reg615		; <i16> [#uses=1]
	%cast620 = sext i16 %reg137 to i32		; <i32> [#uses=2]
	%cond621 = icmp eq i32 %cast620, 0		; <i1> [#uses=1]
	br i1 %cond621, label %bb16, label %bb9

bb9:		; preds = %bb8
	%reg139 = load i32* @yychar		; <i32> [#uses=2]
	%reg460 = add i32 %cast620, %reg139		; <i32> [#uses=3]
	%cond624 = icmp slt i32 %reg460, 0		; <i1> [#uses=1]
	br i1 %cond624, label %bb16, label %bb10

bb10:		; preds = %bb9
	%cond627 = icmp sgt i32 %reg460, 262		; <i1> [#uses=1]
	br i1 %cond627, label %bb16, label %bb11

bb11:		; preds = %bb10
	%reg460-idxcast = sext i32 %reg460 to i64		; <i64> [#uses=2]
	%reg632 = getelementptr [263 x i16]* @yycheck, i64 0, i64 %reg460-idxcast		; <i16*> [#uses=1]
	%reg148 = load i16* %reg632		; <i16> [#uses=1]
	%cast637 = sext i16 %reg148 to i32		; <i32> [#uses=1]
	%cond639 = icmp ne i32 %cast637, %reg139		; <i1> [#uses=1]
	br i1 %cond639, label %bb16, label %bb12

bb12:		; preds = %bb11
	%reg150 = load i16** @yyssp		; <i16*> [#uses=1]
	%cast640 = bitcast i16* %reg150 to i8*		; <i8*> [#uses=1]
	%reg151 = load i16** @yysslim		; <i16*> [#uses=1]
	%cast641 = bitcast i16* %reg151 to i8*		; <i8*> [#uses=1]
	%cond642 = icmp ult i8* %cast640, %cast641		; <i1> [#uses=1]
	br i1 %cond642, label %bb14, label %bb13

bb13:		; preds = %bb12
	%reg644 = call i32 @yygrowstack( )		; <i32> [#uses=1]
	%cond644 = icmp ne i32 %reg644, 0		; <i1> [#uses=1]
	br i1 %cond644, label %bb113, label %bb14

bb14:		; preds = %bb13, %bb12
	%reg153 = load i16** @yyssp		; <i16*> [#uses=1]
	%reg647 = getelementptr i16* %reg153, i64 1		; <i16*> [#uses=2]
	store i16* %reg647, i16** @yyssp
	%reg653 = getelementptr [263 x i16]* @yytable, i64 0, i64 %reg460-idxcast		; <i16*> [#uses=1]
	%reg162 = load i16* %reg653		; <i16> [#uses=2]
	%cast658 = sext i16 %reg162 to i32		; <i32> [#uses=2]
	store i16 %reg162, i16* %reg647
	%reg164 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg661 = getelementptr %YYSTYPE* %reg164, i64 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg661, %YYSTYPE** @yyvsp
	%reg167 = load %IntList* getelementptr (%YYSTYPE* @yylval, i64 0, i32 0)		; <%IntList> [#uses=1]
	%reg661.idx1 = getelementptr %YYSTYPE* %reg164, i64 1, i32 0		; <%IntList*> [#uses=1]
	store %IntList %reg167, %IntList* %reg661.idx1
	store i32 -1, i32* @yychar
	%reg169 = load i32* @yyerrflag		; <i32> [#uses=2]
	%cond669 = icmp sle i32 %reg169, 0		; <i1> [#uses=1]
	br i1 %cond669, label %bb4, label %bb15

bb15:		; preds = %bb14
	%reg171 = add i32 %reg169, -1		; <i32> [#uses=1]
	store i32 %reg171, i32* @yyerrflag
	br label %bb4

bb16:		; preds = %bb11, %bb10, %bb9, %bb8
	%reg677 = getelementptr [43 x i16]* @yyrindex, i64 0, i64 %reg458-idxcast		; <i16*> [#uses=1]
	%reg178 = load i16* %reg677		; <i16> [#uses=1]
	%cast682 = sext i16 %reg178 to i32		; <i32> [#uses=2]
	%cond683 = icmp eq i32 %cast682, 0		; <i1> [#uses=1]
	br i1 %cond683, label %bb21, label %bb17

bb17:		; preds = %bb16
	%reg180 = load i32* @yychar		; <i32> [#uses=2]
	%reg463 = add i32 %cast682, %reg180		; <i32> [#uses=3]
	%cond686 = icmp slt i32 %reg463, 0		; <i1> [#uses=1]
	br i1 %cond686, label %bb21, label %bb18

bb18:		; preds = %bb17
	%cond689 = icmp sgt i32 %reg463, 262		; <i1> [#uses=1]
	br i1 %cond689, label %bb21, label %bb19

bb19:		; preds = %bb18
	%reg463-idxcast = sext i32 %reg463 to i64		; <i64> [#uses=2]
	%reg694 = getelementptr [263 x i16]* @yycheck, i64 0, i64 %reg463-idxcast		; <i16*> [#uses=1]
	%reg189 = load i16* %reg694		; <i16> [#uses=1]
	%cast699 = sext i16 %reg189 to i32		; <i32> [#uses=1]
	%cond701 = icmp ne i32 %cast699, %reg180		; <i1> [#uses=1]
	br i1 %cond701, label %bb21, label %bb20

bb20:		; preds = %bb19
	%reg704 = getelementptr [263 x i16]* @yytable, i64 0, i64 %reg463-idxcast		; <i16*> [#uses=1]
	%reg197 = load i16* %reg704		; <i16> [#uses=1]
	%cast709 = sext i16 %reg197 to i32		; <i32> [#uses=1]
	br label %bb36

bb21:		; preds = %bb19, %bb18, %bb17, %bb16
	%reg198 = load i32* @yyerrflag		; <i32> [#uses=1]
	%cond711 = icmp ne i32 %reg198, 0		; <i1> [#uses=1]
	br i1 %cond711, label %bb23, label %bb22

bb22:		; preds = %bb21
	call void @yyerror( i8* getelementptr ([13 x i8]* @.LC01, i64 0, i64 0) )
	%reg200 = load i32* @yynerrs		; <i32> [#uses=1]
	%reg201 = add i32 %reg200, 1		; <i32> [#uses=1]
	store i32 %reg201, i32* @yynerrs
	br label %bb23

bb23:		; preds = %bb22, %bb21
	%reg202 = load i32* @yyerrflag		; <i32> [#uses=1]
	%cond719 = icmp sgt i32 %reg202, 2		; <i1> [#uses=1]
	br i1 %cond719, label %bb34, label %bb24

bb24:		; preds = %bb23
	store i32 3, i32* @yyerrflag
	%reg241 = load i16** @yyss		; <i16*> [#uses=1]
	%cast778 = bitcast i16* %reg241 to i8*		; <i8*> [#uses=1]
	br label %bb25

bb25:		; preds = %bb33, %bb24
	%reg204 = load i16** @yyssp		; <i16*> [#uses=4]
	%reg206 = load i16* %reg204		; <i16> [#uses=1]
	%reg206-idxcast = sext i16 %reg206 to i64		; <i64> [#uses=1]
	%reg727 = getelementptr [43 x i16]* @yysindex, i64 0, i64 %reg206-idxcast		; <i16*> [#uses=1]
	%reg212 = load i16* %reg727		; <i16> [#uses=2]
	%cast732 = sext i16 %reg212 to i32		; <i32> [#uses=2]
	%cond733 = icmp eq i32 %cast732, 0		; <i1> [#uses=1]
	br i1 %cond733, label %bb32, label %bb26

bb26:		; preds = %bb25
	%reg466 = add i32 %cast732, 256		; <i32> [#uses=2]
	%cond736 = icmp slt i32 %reg466, 0		; <i1> [#uses=1]
	br i1 %cond736, label %bb32, label %bb27

bb27:		; preds = %bb26
	%cond739 = icmp sgt i32 %reg466, 262		; <i1> [#uses=1]
	br i1 %cond739, label %bb32, label %bb28

bb28:		; preds = %bb27
	%reg212-idxcast = sext i16 %reg212 to i64		; <i64> [#uses=1]
	%reg212-idxcast-offset = add i64 %reg212-idxcast, 256		; <i64> [#uses=2]
	%reg744 = getelementptr [263 x i16]* @yycheck, i64 0, i64 %reg212-idxcast-offset		; <i16*> [#uses=1]
	%reg221 = load i16* %reg744		; <i16> [#uses=1]
	%cond748 = icmp ne i16 %reg221, 256		; <i1> [#uses=1]
	br i1 %cond748, label %bb32, label %bb29

bb29:		; preds = %bb28
	%cast750 = bitcast i16* %reg204 to i8*		; <i8*> [#uses=1]
	%reg223 = load i16** @yysslim		; <i16*> [#uses=1]
	%cast751 = bitcast i16* %reg223 to i8*		; <i8*> [#uses=1]
	%cond752 = icmp ult i8* %cast750, %cast751		; <i1> [#uses=1]
	br i1 %cond752, label %bb31, label %bb30

bb30:		; preds = %bb29
	%reg754 = call i32 @yygrowstack( )		; <i32> [#uses=1]
	%cond754 = icmp ne i32 %reg754, 0		; <i1> [#uses=1]
	br i1 %cond754, label %bb113, label %bb31

bb31:		; preds = %bb30, %bb29
	%reg225 = load i16** @yyssp		; <i16*> [#uses=1]
	%reg757 = getelementptr i16* %reg225, i64 1		; <i16*> [#uses=2]
	store i16* %reg757, i16** @yyssp
	%reg763 = getelementptr [263 x i16]* @yytable, i64 0, i64 %reg212-idxcast-offset		; <i16*> [#uses=1]
	%reg234 = load i16* %reg763		; <i16> [#uses=2]
	%cast768 = sext i16 %reg234 to i32		; <i32> [#uses=1]
	store i16 %reg234, i16* %reg757
	%reg236 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg771 = getelementptr %YYSTYPE* %reg236, i64 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg771, %YYSTYPE** @yyvsp
	%reg239 = load %IntList* getelementptr (%YYSTYPE* @yylval, i64 0, i32 0)		; <%IntList> [#uses=1]
	%reg771.idx1 = getelementptr %YYSTYPE* %reg236, i64 1, i32 0		; <%IntList*> [#uses=1]
	store %IntList %reg239, %IntList* %reg771.idx1
	br label %bb4

bb32:		; preds = %bb28, %bb27, %bb26, %bb25
	%cast777 = bitcast i16* %reg204 to i8*		; <i8*> [#uses=1]
	%cond779 = icmp ule i8* %cast777, %cast778		; <i1> [#uses=1]
	br i1 %cond779, label %UnifiedExitNode, label %bb33

bb33:		; preds = %bb32
	%reg781 = getelementptr i16* %reg204, i64 -1		; <i16*> [#uses=1]
	store i16* %reg781, i16** @yyssp
	%reg244 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%reg786 = getelementptr %YYSTYPE* %reg244, i64 -1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg786, %YYSTYPE** @yyvsp
	br label %bb25

bb34:		; preds = %bb23
	%reg246 = load i32* @yychar		; <i32> [#uses=1]
	%cond791 = icmp eq i32 %reg246, 0		; <i1> [#uses=1]
	br i1 %cond791, label %UnifiedExitNode, label %bb35

bb35:		; preds = %bb34
	store i32 -1, i32* @yychar
	br label %bb4

bb36:		; preds = %bb20, %bb4
	%reg468 = phi i32 [ %cast709, %bb20 ], [ %cast599, %bb4 ]		; <i32> [#uses=31]
	%reg468-idxcast = sext i32 %reg468 to i64		; <i64> [#uses=2]
	%reg796 = getelementptr [25 x i16]* @yylen, i64 0, i64 %reg468-idxcast		; <i16*> [#uses=1]
	%reg254 = load i16* %reg796		; <i16> [#uses=2]
	%reg259 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%reg254-idxcast = sext i16 %reg254 to i64		; <i64> [#uses=1]
	%reg254-idxcast-scale = mul i64 %reg254-idxcast, -1		; <i64> [#uses=1]
	%reg254-idxcast-scale-offset = add i64 %reg254-idxcast-scale, 1		; <i64> [#uses=1]
	%reg261.idx1 = getelementptr %YYSTYPE* %reg259, i64 %reg254-idxcast-scale-offset, i32 0		; <%IntList*> [#uses=1]
	%reg261 = load %IntList* %reg261.idx1		; <%IntList> [#uses=1]
	store %IntList %reg261, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	%cond812 = icmp eq i32 %reg468, 13		; <i1> [#uses=1]
	br i1 %cond812, label %bb85, label %bb37

bb37:		; preds = %bb36
	%cond814 = icmp sgt i32 %reg468, 13		; <i1> [#uses=1]
	br i1 %cond814, label %bb56, label %bb38

bb38:		; preds = %bb37
	%cond817 = icmp eq i32 %reg468, 7		; <i1> [#uses=1]
	br i1 %cond817, label %bb79, label %bb39

bb39:		; preds = %bb38
	%cond819 = icmp sgt i32 %reg468, 7		; <i1> [#uses=1]
	br i1 %cond819, label %bb48, label %bb40

bb40:		; preds = %bb39
	%cond822 = icmp eq i32 %reg468, 4		; <i1> [#uses=1]
	br i1 %cond822, label %bb76, label %bb41

bb41:		; preds = %bb40
	%cond824 = icmp sgt i32 %reg468, 4		; <i1> [#uses=1]
	br i1 %cond824, label %bb45, label %bb42

bb42:		; preds = %bb41
	%cond827 = icmp eq i32 %reg468, 2		; <i1> [#uses=1]
	br i1 %cond827, label %bb74, label %bb43

bb43:		; preds = %bb42
	%cond829 = icmp eq i32 %reg468, 3		; <i1> [#uses=1]
	br i1 %cond829, label %bb75, label %bb97

bb45:		; preds = %bb41
	%cond831 = icmp eq i32 %reg468, 5		; <i1> [#uses=1]
	br i1 %cond831, label %bb77, label %bb46

bb46:		; preds = %bb45
	%cond833 = icmp eq i32 %reg468, 6		; <i1> [#uses=1]
	br i1 %cond833, label %bb78, label %bb97

bb48:		; preds = %bb39
	%cond835 = icmp eq i32 %reg468, 10		; <i1> [#uses=1]
	br i1 %cond835, label %bb82, label %bb49

bb49:		; preds = %bb48
	%cond837 = icmp sgt i32 %reg468, 10		; <i1> [#uses=1]
	br i1 %cond837, label %bb53, label %bb50

bb50:		; preds = %bb49
	%cond840 = icmp eq i32 %reg468, 8		; <i1> [#uses=1]
	br i1 %cond840, label %bb80, label %bb51

bb51:		; preds = %bb50
	%cond842 = icmp eq i32 %reg468, 9		; <i1> [#uses=1]
	br i1 %cond842, label %bb81, label %bb97

bb53:		; preds = %bb49
	%cond844 = icmp eq i32 %reg468, 11		; <i1> [#uses=1]
	br i1 %cond844, label %bb83, label %bb54

bb54:		; preds = %bb53
	%cond846 = icmp eq i32 %reg468, 12		; <i1> [#uses=1]
	br i1 %cond846, label %bb84, label %bb97

bb56:		; preds = %bb37
	%cond848 = icmp eq i32 %reg468, 19		; <i1> [#uses=1]
	br i1 %cond848, label %bb91, label %bb57

bb57:		; preds = %bb56
	%cond850 = icmp sgt i32 %reg468, 19		; <i1> [#uses=1]
	br i1 %cond850, label %bb66, label %bb58

bb58:		; preds = %bb57
	%cond853 = icmp eq i32 %reg468, 16		; <i1> [#uses=1]
	br i1 %cond853, label %bb88, label %bb59

bb59:		; preds = %bb58
	%cond855 = icmp sgt i32 %reg468, 16		; <i1> [#uses=1]
	br i1 %cond855, label %bb63, label %bb60

bb60:		; preds = %bb59
	%cond858 = icmp eq i32 %reg468, 14		; <i1> [#uses=1]
	br i1 %cond858, label %bb86, label %bb61

bb61:		; preds = %bb60
	%cond860 = icmp eq i32 %reg468, 15		; <i1> [#uses=1]
	br i1 %cond860, label %bb87, label %bb97

bb63:		; preds = %bb59
	%cond862 = icmp eq i32 %reg468, 17		; <i1> [#uses=1]
	br i1 %cond862, label %bb89, label %bb64

bb64:		; preds = %bb63
	%cond864 = icmp eq i32 %reg468, 18		; <i1> [#uses=1]
	br i1 %cond864, label %bb90, label %bb97

bb66:		; preds = %bb57
	%cond866 = icmp eq i32 %reg468, 22		; <i1> [#uses=1]
	br i1 %cond866, label %bb94, label %bb67

bb67:		; preds = %bb66
	%cond868 = icmp sgt i32 %reg468, 22		; <i1> [#uses=1]
	br i1 %cond868, label %bb71, label %bb68

bb68:		; preds = %bb67
	%cond871 = icmp eq i32 %reg468, 20		; <i1> [#uses=1]
	br i1 %cond871, label %bb92, label %bb69

bb69:		; preds = %bb68
	%cond873 = icmp eq i32 %reg468, 21		; <i1> [#uses=1]
	br i1 %cond873, label %bb93, label %bb97

bb71:		; preds = %bb67
	%cond875 = icmp eq i32 %reg468, 23		; <i1> [#uses=1]
	br i1 %cond875, label %bb95, label %bb72

bb72:		; preds = %bb71
	%cond877 = icmp eq i32 %reg468, 24		; <i1> [#uses=1]
	br i1 %cond877, label %bb96, label %bb97

bb74:		; preds = %bb42
	call void @yyfinished( )
	br label %bb97

bb75:		; preds = %bb43
	%reg262 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg264.idx1 = getelementptr %YYSTYPE* %reg262, i64 -2, i32 0		; <%IntList*> [#uses=1]
	%reg264 = load %IntList* %reg264.idx1		; <%IntList> [#uses=1]
	%reg265.idx = getelementptr %YYSTYPE* %reg262, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg265 = load %IntList* %reg265.idx		; <%IntList> [#uses=1]
	%cast889 = bitcast %IntList %reg265 to %List		; <%List> [#uses=1]
	%cast890 = bitcast %IntList %reg264 to %List		; <%List> [#uses=1]
	call void @doSpec( %List %cast890, %List %cast889 )
	br label %bb97

bb76:		; preds = %bb40
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb77:		; preds = %bb45
	%reg269 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast894 = getelementptr %YYSTYPE* %reg269, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg271 = load %IntList* %cast894		; <%IntList> [#uses=1]
	%reg271.upgrd.1 = bitcast %IntList %reg271 to i8*		; <i8*> [#uses=1]
	%reg272.idx1 = getelementptr %YYSTYPE* %reg269, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg272 = load %IntList* %reg272.idx1		; <%IntList> [#uses=1]
	%cast901 = bitcast %IntList %reg272 to %List		; <%List> [#uses=1]
	%reg901 = call %List @newList( i8* %reg271.upgrd.1, %List %cast901 )		; <%List> [#uses=1]
	bitcast %List %reg901 to %IntList		; <%IntList>:0 [#uses=1]
	store %IntList %0, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb78:		; preds = %bb46
	%reg275 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%reg277.idx = getelementptr %YYSTYPE* %reg275, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg277 = load %IntList* %reg277.idx		; <%IntList> [#uses=1]
	%cast907 = bitcast %IntList %reg277 to %List		; <%List> [#uses=1]
	%reg907 = call %Arity @newArity( i32 -1, %List %cast907 )		; <%Arity> [#uses=1]
	bitcast %Arity %reg907 to %IntList		; <%IntList>:1 [#uses=1]
	store %IntList %1, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb79:		; preds = %bb38
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	%reg281 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast912 = getelementptr %YYSTYPE* %reg281, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg282 = load %IntList* %cast912		; <%IntList> [#uses=1]
	%reg282.upgrd.2 = bitcast %IntList %reg282 to %List		; <%List> [#uses=1]
	call void @doGram( %List %reg282.upgrd.2 )
	br label %bb97

bb80:		; preds = %bb50
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	%reg285 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast917 = getelementptr %YYSTYPE* %reg285, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg286 = load %IntList* %cast917		; <%IntList> [#uses=1]
	%reg286.upgrd.3 = bitcast %IntList %reg286 to i8*		; <i8*> [#uses=1]
	call void @doStart( i8* %reg286.upgrd.3 )
	br label %bb97

bb81:		; preds = %bb51
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb82:		; preds = %bb48
	%reg290 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast923 = getelementptr %YYSTYPE* %reg290, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg292 = load %IntList* %cast923		; <%IntList> [#uses=1]
	%reg292.upgrd.4 = bitcast %IntList %reg292 to i8*		; <i8*> [#uses=1]
	%reg293.idx1 = getelementptr %YYSTYPE* %reg290, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg293 = load %IntList* %reg293.idx1		; <%IntList> [#uses=1]
	%cast930 = bitcast %IntList %reg293 to %List		; <%List> [#uses=1]
	%reg930 = call %List @newList( i8* %reg292.upgrd.4, %List %cast930 )		; <%List> [#uses=1]
	bitcast %List %reg930 to %IntList		; <%IntList>:2 [#uses=1]
	store %IntList %2, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb83:		; preds = %bb53
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb84:		; preds = %bb54
	%reg298 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast936 = getelementptr %YYSTYPE* %reg298, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg300 = load %IntList* %cast936		; <%IntList> [#uses=1]
	%reg300.upgrd.5 = bitcast %IntList %reg300 to i8*		; <i8*> [#uses=1]
	%reg301.idx1 = getelementptr %YYSTYPE* %reg298, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg301 = load %IntList* %reg301.idx1		; <%IntList> [#uses=1]
	%cast943 = bitcast %IntList %reg301 to %List		; <%List> [#uses=1]
	%reg943 = call %List @newList( i8* %reg300.upgrd.5, %List %cast943 )		; <%List> [#uses=1]
	bitcast %List %reg943 to %IntList		; <%IntList>:3 [#uses=1]
	store %IntList %3, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb85:		; preds = %bb36
	%reg304 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast9521 = getelementptr %YYSTYPE* %reg304, i64 -2, i32 0		; <%IntList*> [#uses=1]
	%reg306 = load %IntList* %cast9521		; <%IntList> [#uses=1]
	%reg306.upgrd.6 = bitcast %IntList %reg306 to i8*		; <i8*> [#uses=1]
	%cast953 = bitcast %YYSTYPE* %reg304 to i32*		; <i32*> [#uses=1]
	%reg307 = load i32* %cast953		; <i32> [#uses=1]
	%reg955 = call %Binding @newBinding( i8* %reg306.upgrd.6, i32 %reg307 )		; <%Binding> [#uses=1]
	bitcast %Binding %reg955 to %IntList		; <%IntList>:4 [#uses=1]
	store %IntList %4, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb86:		; preds = %bb60
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb87:		; preds = %bb61
	%reg312 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast961 = getelementptr %YYSTYPE* %reg312, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg314 = load %IntList* %cast961		; <%IntList> [#uses=1]
	%reg314.upgrd.7 = bitcast %IntList %reg314 to i8*		; <i8*> [#uses=1]
	%reg315.idx1 = getelementptr %YYSTYPE* %reg312, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg315 = load %IntList* %reg315.idx1		; <%IntList> [#uses=1]
	%cast968 = bitcast %IntList %reg315 to %List		; <%List> [#uses=1]
	%reg968 = call %List @newList( i8* %reg314.upgrd.7, %List %cast968 )		; <%List> [#uses=1]
	bitcast %List %reg968 to %IntList		; <%IntList>:5 [#uses=1]
	store %IntList %5, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb88:		; preds = %bb58
	%reg318 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=4]
	%cast9791 = getelementptr %YYSTYPE* %reg318, i64 -6, i32 0		; <%IntList*> [#uses=1]
	%reg322 = load %IntList* %cast9791		; <%IntList> [#uses=1]
	%reg322.upgrd.8 = bitcast %IntList %reg322 to i8*		; <i8*> [#uses=1]
	%reg323.idx1 = getelementptr %YYSTYPE* %reg318, i64 -4, i32 0		; <%IntList*> [#uses=1]
	%reg323 = load %IntList* %reg323.idx1		; <%IntList> [#uses=1]
	%reg987 = getelementptr %YYSTYPE* %reg318, i64 -2		; <%YYSTYPE*> [#uses=1]
	%cast989 = bitcast %YYSTYPE* %reg987 to i32*		; <i32*> [#uses=1]
	%reg324 = load i32* %cast989		; <i32> [#uses=1]
	%reg325.idx1 = getelementptr %YYSTYPE* %reg318, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg325 = load %IntList* %reg325.idx1		; <%IntList> [#uses=1]
	%cast998 = bitcast %IntList %reg323 to %PatternAST		; <%PatternAST> [#uses=1]
	%reg996 = call %RuleAST @newRuleAST( i8* %reg322.upgrd.8, %PatternAST %cast998, i32 %reg324, %IntList %reg325 )		; <%RuleAST> [#uses=1]
	bitcast %RuleAST %reg996 to %IntList		; <%IntList>:6 [#uses=1]
	store %IntList %6, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb89:		; preds = %bb63
	%reg328 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast1002 = getelementptr %YYSTYPE* %reg328, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg329 = load %IntList* %cast1002		; <%IntList> [#uses=1]
	%reg329.upgrd.9 = bitcast %IntList %reg329 to i8*		; <i8*> [#uses=1]
	%reg1004 = call %PatternAST @newPatternAST( i8* %reg329.upgrd.9, %List null )		; <%PatternAST> [#uses=1]
	bitcast %PatternAST %reg1004 to %IntList		; <%IntList>:7 [#uses=1]
	store %IntList %7, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb90:		; preds = %bb64
	%reg333 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast10131 = getelementptr %YYSTYPE* %reg333, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg335 = load %IntList* %cast10131		; <%IntList> [#uses=1]
	%reg335.upgrd.10 = bitcast %IntList %reg335 to i8*		; <i8*> [#uses=1]
	%reg1015 = call %List @newList( i8* %reg335.upgrd.10, %List null )		; <%List> [#uses=1]
	%cast10211 = getelementptr %YYSTYPE* %reg333, i64 -3, i32 0		; <%IntList*> [#uses=1]
	%reg338 = load %IntList* %cast10211		; <%IntList> [#uses=1]
	%reg338.upgrd.11 = bitcast %IntList %reg338 to i8*		; <i8*> [#uses=1]
	%reg1023 = call %PatternAST @newPatternAST( i8* %reg338.upgrd.11, %List %reg1015 )		; <%PatternAST> [#uses=1]
	bitcast %PatternAST %reg1023 to %IntList		; <%IntList>:8 [#uses=1]
	store %IntList %8, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb91:		; preds = %bb56
	%reg341 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=3]
	%cast10331 = getelementptr %YYSTYPE* %reg341, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg344 = load %IntList* %cast10331		; <%IntList> [#uses=1]
	%reg344.upgrd.12 = bitcast %IntList %reg344 to i8*		; <i8*> [#uses=1]
	%reg1035 = call %List @newList( i8* %reg344.upgrd.12, %List null )		; <%List> [#uses=1]
	%cast10411 = getelementptr %YYSTYPE* %reg341, i64 -3, i32 0		; <%IntList*> [#uses=1]
	%reg347 = load %IntList* %cast10411		; <%IntList> [#uses=1]
	%reg347.upgrd.13 = bitcast %IntList %reg347 to i8*		; <i8*> [#uses=1]
	%reg1043 = call %List @newList( i8* %reg347.upgrd.13, %List %reg1035 )		; <%List> [#uses=1]
	%cast10491 = getelementptr %YYSTYPE* %reg341, i64 -5, i32 0		; <%IntList*> [#uses=1]
	%reg349 = load %IntList* %cast10491		; <%IntList> [#uses=1]
	%reg349.upgrd.14 = bitcast %IntList %reg349 to i8*		; <i8*> [#uses=1]
	%reg1051 = call %PatternAST @newPatternAST( i8* %reg349.upgrd.14, %List %reg1043 )		; <%PatternAST> [#uses=1]
	bitcast %PatternAST %reg1051 to %IntList		; <%IntList>:9 [#uses=1]
	store %IntList %9, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb92:		; preds = %bb68
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb93:		; preds = %bb69
	%reg354 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1059 = getelementptr %YYSTYPE* %reg354, i64 -2		; <%YYSTYPE*> [#uses=1]
	%cast1061 = bitcast %YYSTYPE* %reg1059 to i32*		; <i32*> [#uses=1]
	%reg356 = load i32* %cast1061		; <i32> [#uses=1]
	%reg357.idx1 = getelementptr %YYSTYPE* %reg354, i64 -1, i32 0		; <%IntList*> [#uses=1]
	%reg357 = load %IntList* %reg357.idx1		; <%IntList> [#uses=1]
	%reg1068 = call %IntList @newIntList( i32 %reg356, %IntList %reg357 )		; <%IntList> [#uses=1]
	store %IntList %reg1068, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb94:		; preds = %bb66
	store %IntList null, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb95:		; preds = %bb71
	%reg362 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1076 = getelementptr %YYSTYPE* %reg362, i64 -1		; <%YYSTYPE*> [#uses=1]
	%cast1078 = bitcast %YYSTYPE* %reg1076 to i32*		; <i32*> [#uses=1]
	%reg364 = load i32* %cast1078		; <i32> [#uses=1]
	%reg365.idx = getelementptr %YYSTYPE* %reg362, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg365 = load %IntList* %reg365.idx		; <%IntList> [#uses=1]
	%reg1081 = call %IntList @newIntList( i32 %reg364, %IntList %reg365 )		; <%IntList> [#uses=1]
	store %IntList %reg1081, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb96:		; preds = %bb72
	%reg368 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1088 = getelementptr %YYSTYPE* %reg368, i64 -1		; <%YYSTYPE*> [#uses=1]
	%cast1090 = bitcast %YYSTYPE* %reg1088 to i32*		; <i32*> [#uses=1]
	%reg370 = load i32* %cast1090		; <i32> [#uses=1]
	%reg371.idx = getelementptr %YYSTYPE* %reg368, i64 0, i32 0		; <%IntList*> [#uses=1]
	%reg371 = load %IntList* %reg371.idx		; <%IntList> [#uses=1]
	%reg1093 = call %IntList @newIntList( i32 %reg370, %IntList %reg371 )		; <%IntList> [#uses=1]
	store %IntList %reg1093, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	br label %bb97

bb97:		; preds = %bb96, %bb95, %bb94, %bb93, %bb92, %bb91, %bb90, %bb89, %bb88, %bb87, %bb86, %bb85, %bb84, %bb83, %bb82, %bb81, %bb80, %bb79, %bb78, %bb77, %bb76, %bb75, %bb74, %bb72, %bb69, %bb64, %bb61, %bb54, %bb51, %bb46, %bb43
	%cast1097 = sext i16 %reg254 to i64		; <i64> [#uses=3]
	%reg375 = add i64 %cast1097, %cast1097		; <i64> [#uses=1]
	%reg377 = load i16** @yyssp		; <i16*> [#uses=1]
	%cast379 = ptrtoint i16* %reg377 to i64		; <i64> [#uses=1]
	%reg381 = sub i64 %cast379, %reg375		; <i64> [#uses=1]
	%cast1099 = inttoptr i64 %reg381 to i16*		; <i16*> [#uses=1]
	store i16* %cast1099, i16** @yyssp
	%reg382 = load i16** @yyssp		; <i16*> [#uses=3]
	%reg383 = load i16* %reg382		; <i16> [#uses=1]
	%cast1103 = sext i16 %reg383 to i32		; <i32> [#uses=3]
	%reg385 = mul i64 %cast1097, 8		; <i64> [#uses=1]
	%reg387 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast389 = ptrtoint %YYSTYPE* %reg387 to i64		; <i64> [#uses=1]
	%reg391 = sub i64 %cast389, %reg385		; <i64> [#uses=1]
	%cast1108 = inttoptr i64 %reg391 to %YYSTYPE*		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %cast1108, %YYSTYPE** @yyvsp
	%reg1111 = getelementptr [25 x i16]* @yylhs, i64 0, i64 %reg468-idxcast		; <i16*> [#uses=1]
	%reg398 = load i16* %reg1111		; <i16> [#uses=2]
	%cast1116 = sext i16 %reg398 to i32		; <i32> [#uses=1]
	%cond1117 = icmp ne i32 %cast1103, 0		; <i1> [#uses=1]
	br i1 %cond1117, label %bb104, label %bb98

bb98:		; preds = %bb97
	%cond1119 = icmp ne i32 %cast1116, 0		; <i1> [#uses=1]
	br i1 %cond1119, label %bb104, label %bb99

bb99:		; preds = %bb98
	%reg1122 = getelementptr i16* %reg382, i64 1		; <i16*> [#uses=2]
	store i16* %reg1122, i16** @yyssp
	store i16 1, i16* %reg1122
	%reg403 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1128 = getelementptr %YYSTYPE* %reg403, i64 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg1128, %YYSTYPE** @yyvsp
	%reg406 = load %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)		; <%IntList> [#uses=1]
	%reg1128.idx1 = getelementptr %YYSTYPE* %reg403, i64 1, i32 0		; <%IntList*> [#uses=1]
	store %IntList %reg406, %IntList* %reg1128.idx1
	%reg407 = load i32* @yychar		; <i32> [#uses=1]
	%cond1135 = icmp sge i32 %reg407, 0		; <i1> [#uses=1]
	br i1 %cond1135, label %bb102, label %bb100

bb100:		; preds = %bb99
	%reg1139 = call i32 @yylex( )		; <i32> [#uses=1]
	store i32 %reg1139, i32* @yychar
	%reg409 = load i32* @yychar		; <i32> [#uses=1]
	%cond1141 = icmp sge i32 %reg409, 0		; <i1> [#uses=1]
	br i1 %cond1141, label %bb102, label %bb101

bb101:		; preds = %bb100
	store i32 0, i32* @yychar
	br label %bb102

bb102:		; preds = %bb101, %bb100, %bb99
	%reg411 = load i32* @yychar		; <i32> [#uses=1]
	%cond1146 = icmp ne i32 %reg411, 0		; <i1> [#uses=1]
	br i1 %cond1146, label %bb4, label %UnifiedExitNode

bb104:		; preds = %bb98, %bb97
	%reg398-idxcast = sext i16 %reg398 to i64		; <i64> [#uses=2]
	%reg1150 = getelementptr [12 x i16]* @yygindex, i64 0, i64 %reg398-idxcast		; <i16*> [#uses=1]
	%reg418 = load i16* %reg1150		; <i16> [#uses=1]
	%cast1155 = sext i16 %reg418 to i32		; <i32> [#uses=2]
	%cond1156 = icmp eq i32 %cast1155, 0		; <i1> [#uses=1]
	br i1 %cond1156, label %bb109, label %bb105

bb105:		; preds = %bb104
	%reg473 = add i32 %cast1155, %cast1103		; <i32> [#uses=3]
	%cond1158 = icmp slt i32 %reg473, 0		; <i1> [#uses=1]
	br i1 %cond1158, label %bb109, label %bb106

bb106:		; preds = %bb105
	%cond1161 = icmp sgt i32 %reg473, 262		; <i1> [#uses=1]
	br i1 %cond1161, label %bb109, label %bb107

bb107:		; preds = %bb106
	%reg473-idxcast = sext i32 %reg473 to i64		; <i64> [#uses=2]
	%reg1166 = getelementptr [263 x i16]* @yycheck, i64 0, i64 %reg473-idxcast		; <i16*> [#uses=1]
	%reg428 = load i16* %reg1166		; <i16> [#uses=1]
	%cast1171 = sext i16 %reg428 to i32		; <i32> [#uses=1]
	%cond1172 = icmp ne i32 %cast1171, %cast1103		; <i1> [#uses=1]
	br i1 %cond1172, label %bb109, label %bb108

bb108:		; preds = %bb107
	%reg1175 = getelementptr [263 x i16]* @yytable, i64 0, i64 %reg473-idxcast		; <i16*> [#uses=1]
	%reg435 = load i16* %reg1175		; <i16> [#uses=1]
	%cast1180 = sext i16 %reg435 to i32		; <i32> [#uses=1]
	br label %bb110

bb109:		; preds = %bb107, %bb106, %bb105, %bb104
	%reg1183 = getelementptr [12 x i16]* @yydgoto, i64 0, i64 %reg398-idxcast		; <i16*> [#uses=1]
	%reg442 = load i16* %reg1183		; <i16> [#uses=1]
	%cast1188 = sext i16 %reg442 to i32		; <i32> [#uses=1]
	br label %bb110

bb110:		; preds = %bb109, %bb108
	%reg476 = phi i32 [ %cast1188, %bb109 ], [ %cast1180, %bb108 ]		; <i32> [#uses=2]
	%cast1189 = bitcast i16* %reg382 to i8*		; <i8*> [#uses=1]
	%reg444 = load i16** @yysslim		; <i16*> [#uses=1]
	%cast1190 = bitcast i16* %reg444 to i8*		; <i8*> [#uses=1]
	%cond1191 = icmp ult i8* %cast1189, %cast1190		; <i1> [#uses=1]
	br i1 %cond1191, label %bb112, label %bb111

bb111:		; preds = %bb110
	%reg1193 = call i32 @yygrowstack( )		; <i32> [#uses=1]
	%cond1193 = icmp ne i32 %reg1193, 0		; <i1> [#uses=1]
	br i1 %cond1193, label %bb113, label %bb112

bb112:		; preds = %bb111, %bb110
	%reg446 = load i16** @yyssp		; <i16*> [#uses=1]
	%reg1196 = getelementptr i16* %reg446, i64 1		; <i16*> [#uses=2]
	store i16* %reg1196, i16** @yyssp
	%cast1357 = trunc i32 %reg476 to i16		; <i16> [#uses=1]
	store i16 %cast1357, i16* %reg1196
	%reg449 = load %YYSTYPE** @yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1202 = getelementptr %YYSTYPE* %reg449, i64 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg1202, %YYSTYPE** @yyvsp
	%reg452 = load %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)		; <%IntList> [#uses=1]
	%reg1202.idx1 = getelementptr %YYSTYPE* %reg449, i64 1, i32 0		; <%IntList*> [#uses=1]
	store %IntList %reg452, %IntList* %reg1202.idx1
	br label %bb4

bb113:		; preds = %bb111, %bb30, %bb13, %bb2
	call void @yyerror( i8* getelementptr ([20 x i8]* @.LC1, i64 0, i64 0) )
	br label %UnifiedExitNode

UnifiedExitNode:		; preds = %bb113, %bb102, %bb34, %bb32
	%UnifiedRetVal = phi i32 [ 1, %bb113 ], [ 1, %bb34 ], [ 1, %bb32 ], [ 0, %bb102 ]		; <i32> [#uses=1]
	ret i32 %UnifiedRetVal
}

declare %List @newList(i8*, %List)

declare %IntList @newIntList(i32, %IntList)

declare void @doStart(i8*)

declare void @yyerror(i8*)

declare void @doSpec(%List, %List)

declare %Arity @newArity(i32, %List)

declare %Binding @newBinding(i8*, i32)

declare %PatternAST @newPatternAST(i8*, %List)

declare %RuleAST @newRuleAST(i8*, %PatternAST, i32, %IntList)

declare void @yyfinished()

declare i32 @yylex()

declare void @doGram(%List)

declare i32 @yygrowstack()
