; RUN: llc < %s -mtriple=armv6-apple-darwin

  %struct.term = type { i32, i32, i32 }

declare fastcc i8* @memory_Malloc(i32) nounwind

define fastcc %struct.term* @t1() nounwind {
entry:
	br i1 undef, label %bb, label %bb1

bb:		; preds = %entry
	ret %struct.term* undef

bb1:		; preds = %entry
	%0 = tail call fastcc i8* @memory_Malloc(i32 12) nounwind		; <i8*> [#uses=0]
	%1 = tail call fastcc i8* @memory_Malloc(i32 12) nounwind		; <i8*> [#uses=0]
	ret %struct.term* undef
}


define i32 @t2(i32 %argc, i8** nocapture %argv) nounwind {
entry:
	br label %bb6.i8

bb6.i8:		; preds = %memory_CalculateRealBlockSize1374.exit.i, %entry
	br i1 undef, label %memory_CalculateRealBlockSize1374.exit.i, label %bb.i.i9

bb.i.i9:		; preds = %bb6.i8
	br label %memory_CalculateRealBlockSize1374.exit.i

memory_CalculateRealBlockSize1374.exit.i:		; preds = %bb.i.i9, %bb6.i8
	%0 = phi i32 [ undef, %bb.i.i9 ], [ undef, %bb6.i8 ]		; <i32> [#uses=2]
	store i32 %0, i32* undef, align 4
	%1 = urem i32 8184, %0		; <i32> [#uses=1]
	%2 = sub i32 8188, %1		; <i32> [#uses=1]
	store i32 %2, i32* undef, align 4
	br i1 undef, label %memory_Init.exit, label %bb6.i8

memory_Init.exit:		; preds = %memory_CalculateRealBlockSize1374.exit.i
	br label %bb.i.i

bb.i.i:		; preds = %bb.i.i, %memory_Init.exit
	br i1 undef, label %symbol_Init.exit, label %bb.i.i

symbol_Init.exit:		; preds = %bb.i.i
	br label %bb.i.i67

bb.i.i67:		; preds = %bb.i.i67, %symbol_Init.exit
	br i1 undef, label %symbol_CreatePrecedence3522.exit, label %bb.i.i67

symbol_CreatePrecedence3522.exit:		; preds = %bb.i.i67
	br label %bb.i.i8.i

bb.i.i8.i:		; preds = %bb.i.i8.i, %symbol_CreatePrecedence3522.exit
	br i1 undef, label %cont_Create.exit9.i, label %bb.i.i8.i

cont_Create.exit9.i:		; preds = %bb.i.i8.i
	br label %bb.i.i.i72

bb.i.i.i72:		; preds = %bb.i.i.i72, %cont_Create.exit9.i
	br i1 undef, label %cont_Init.exit, label %bb.i.i.i72

cont_Init.exit:		; preds = %bb.i.i.i72
	br label %bb.i103

bb.i103:		; preds = %bb.i103, %cont_Init.exit
	br i1 undef, label %subs_Init.exit, label %bb.i103

subs_Init.exit:		; preds = %bb.i103
	br i1 undef, label %bb1.i.i.i80, label %cc_Init.exit

bb1.i.i.i80:		; preds = %subs_Init.exit
	unreachable

cc_Init.exit:		; preds = %subs_Init.exit
	br label %bb.i.i375

bb.i.i375:		; preds = %bb.i.i375, %cc_Init.exit
	br i1 undef, label %bb.i439, label %bb.i.i375

bb.i439:		; preds = %bb.i439, %bb.i.i375
	br i1 undef, label %opts_DeclareSPASSFlagsAsOptions.exit, label %bb.i439

opts_DeclareSPASSFlagsAsOptions.exit:		; preds = %bb.i439
	br i1 undef, label %opts_TranslateShortOptDeclarations.exit.i, label %bb.i.i82

bb.i.i82:		; preds = %opts_DeclareSPASSFlagsAsOptions.exit
	unreachable

opts_TranslateShortOptDeclarations.exit.i:		; preds = %opts_DeclareSPASSFlagsAsOptions.exit
	br i1 undef, label %list_Length.exit.i.thread.i, label %bb.i.i4.i

list_Length.exit.i.thread.i:		; preds = %opts_TranslateShortOptDeclarations.exit.i
	br i1 undef, label %bb18.i.i.i, label %bb26.i.i.i

bb.i.i4.i:		; preds = %opts_TranslateShortOptDeclarations.exit.i
	unreachable

bb18.i.i.i:		; preds = %list_Length.exit.i.thread.i
	unreachable

bb26.i.i.i:		; preds = %list_Length.exit.i.thread.i
	br i1 undef, label %bb27.i142, label %opts_GetOptLongOnly.exit.thread97.i

opts_GetOptLongOnly.exit.thread97.i:		; preds = %bb26.i.i.i
	br label %bb27.i142

bb27.i142:		; preds = %opts_GetOptLongOnly.exit.thread97.i, %bb26.i.i.i
	br label %bb1.i3.i

bb1.i3.i:		; preds = %bb1.i3.i, %bb27.i142
	br i1 undef, label %opts_FreeLongOptsArray.exit.i, label %bb1.i3.i

opts_FreeLongOptsArray.exit.i:		; preds = %bb1.i3.i
	br label %bb.i443

bb.i443:		; preds = %bb.i443, %opts_FreeLongOptsArray.exit.i
	br i1 undef, label %flag_InitStoreByDefaults3542.exit, label %bb.i443

flag_InitStoreByDefaults3542.exit:		; preds = %bb.i443
	br i1 undef, label %bb6.i449, label %bb.i503

bb6.i449:		; preds = %flag_InitStoreByDefaults3542.exit
	unreachable

bb.i503:		; preds = %bb.i503, %flag_InitStoreByDefaults3542.exit
	br i1 undef, label %flag_CleanStore3464.exit, label %bb.i503

flag_CleanStore3464.exit:		; preds = %bb.i503
	br i1 undef, label %bb1.i81.i.preheader, label %bb.i173

bb.i173:		; preds = %flag_CleanStore3464.exit
	unreachable

bb1.i81.i.preheader:		; preds = %flag_CleanStore3464.exit
	br i1 undef, label %bb1.i64.i.preheader, label %bb5.i179

bb5.i179:		; preds = %bb1.i81.i.preheader
	unreachable

bb1.i64.i.preheader:		; preds = %bb1.i81.i.preheader
	br i1 undef, label %dfg_DeleteProofList.exit.i, label %bb.i9.i

bb.i9.i:		; preds = %bb1.i64.i.preheader
	unreachable

dfg_DeleteProofList.exit.i:		; preds = %bb1.i64.i.preheader
	br i1 undef, label %term_DeleteTermList621.exit.i, label %bb.i.i62.i

bb.i.i62.i:		; preds = %bb.i.i62.i, %dfg_DeleteProofList.exit.i
	br i1 undef, label %term_DeleteTermList621.exit.i, label %bb.i.i62.i

term_DeleteTermList621.exit.i:		; preds = %bb.i.i62.i, %dfg_DeleteProofList.exit.i
	br i1 undef, label %dfg_DFGParser.exit, label %bb.i.i211

bb.i.i211:		; preds = %term_DeleteTermList621.exit.i
	unreachable

dfg_DFGParser.exit:		; preds = %term_DeleteTermList621.exit.i
	br label %bb.i513

bb.i513:		; preds = %bb2.i516, %dfg_DFGParser.exit
	br i1 undef, label %bb2.i516, label %bb1.i514

bb1.i514:		; preds = %bb.i513
	unreachable

bb2.i516:		; preds = %bb.i513
	br i1 undef, label %bb.i509, label %bb.i513

bb.i509:		; preds = %bb.i509, %bb2.i516
	br i1 undef, label %symbol_TransferPrecedence3468.exit511, label %bb.i509

symbol_TransferPrecedence3468.exit511:		; preds = %bb.i509
	br i1 undef, label %bb20, label %bb21

bb20:		; preds = %symbol_TransferPrecedence3468.exit511
	unreachable

bb21:		; preds = %symbol_TransferPrecedence3468.exit511
	br i1 undef, label %cnf_Init.exit, label %bb.i498

bb.i498:		; preds = %bb21
	unreachable

cnf_Init.exit:		; preds = %bb21
	br i1 undef, label %bb23, label %bb22

bb22:		; preds = %cnf_Init.exit
	br i1 undef, label %bb2.i.i496, label %bb.i.i494

bb.i.i494:		; preds = %bb22
	unreachable

bb2.i.i496:		; preds = %bb22
	unreachable

bb23:		; preds = %cnf_Init.exit
	br i1 undef, label %bb28, label %bb24

bb24:		; preds = %bb23
	unreachable

bb28:		; preds = %bb23
	br i1 undef, label %bb31, label %bb29

bb29:		; preds = %bb28
	unreachable

bb31:		; preds = %bb28
	br i1 undef, label %bb34, label %bb32

bb32:		; preds = %bb31
	unreachable

bb34:		; preds = %bb31
	br i1 undef, label %bb83, label %bb66

bb66:		; preds = %bb34
	unreachable

bb83:		; preds = %bb34
	br i1 undef, label %bb2.i1668, label %bb.i1667

bb.i1667:		; preds = %bb83
	unreachable

bb2.i1668:		; preds = %bb83
	br i1 undef, label %bb5.i205, label %bb3.i204

bb3.i204:		; preds = %bb2.i1668
	unreachable

bb5.i205:		; preds = %bb2.i1668
	br i1 undef, label %bb.i206.i, label %ana_AnalyzeSortStructure.exit.i

bb.i206.i:		; preds = %bb5.i205
	br i1 undef, label %bb1.i207.i, label %ana_AnalyzeSortStructure.exit.i

bb1.i207.i:		; preds = %bb.i206.i
	br i1 undef, label %bb25.i1801.thread, label %bb.i1688

bb.i1688:		; preds = %bb1.i207.i
	unreachable

bb25.i1801.thread:		; preds = %bb1.i207.i
	unreachable

ana_AnalyzeSortStructure.exit.i:		; preds = %bb.i206.i, %bb5.i205
	br i1 undef, label %bb7.i207, label %bb.i1806

bb.i1806:		; preds = %ana_AnalyzeSortStructure.exit.i
	br i1 undef, label %bb2.i.i.i1811, label %bb.i.i.i1809

bb.i.i.i1809:		; preds = %bb.i1806
	unreachable

bb2.i.i.i1811:		; preds = %bb.i1806
	unreachable

bb7.i207:		; preds = %ana_AnalyzeSortStructure.exit.i
	br i1 undef, label %bb9.i, label %bb8.i

bb8.i:		; preds = %bb7.i207
	unreachable

bb9.i:		; preds = %bb7.i207
	br i1 undef, label %bb23.i, label %bb26.i

bb23.i:		; preds = %bb9.i
	br i1 undef, label %bb25.i, label %bb24.i

bb24.i:		; preds = %bb23.i
	br i1 undef, label %sort_SortTheoryIsTrivial.exit.i, label %bb.i2093

bb.i2093:		; preds = %bb.i2093, %bb24.i
	br label %bb.i2093

sort_SortTheoryIsTrivial.exit.i:		; preds = %bb24.i
	br i1 undef, label %bb3.i2141, label %bb4.i2143

bb3.i2141:		; preds = %sort_SortTheoryIsTrivial.exit.i
	unreachable

bb4.i2143:		; preds = %sort_SortTheoryIsTrivial.exit.i
	br i1 undef, label %bb8.i2178, label %bb5.i2144

bb5.i2144:		; preds = %bb4.i2143
	br i1 undef, label %bb7.i2177, label %bb1.i28.i

bb1.i28.i:		; preds = %bb5.i2144
	br i1 undef, label %bb4.i43.i, label %bb2.i.i2153

bb2.i.i2153:		; preds = %bb1.i28.i
	br i1 undef, label %bb4.i.i33.i, label %bb.i.i30.i

bb.i.i30.i:		; preds = %bb2.i.i2153
	unreachable

bb4.i.i33.i:		; preds = %bb2.i.i2153
	br i1 undef, label %bb9.i.i36.i, label %bb5.i.i34.i

bb5.i.i34.i:		; preds = %bb4.i.i33.i
	unreachable

bb9.i.i36.i:		; preds = %bb4.i.i33.i
	br i1 undef, label %bb14.i.i.i2163, label %bb10.i.i37.i

bb10.i.i37.i:		; preds = %bb9.i.i36.i
	unreachable

bb14.i.i.i2163:		; preds = %bb9.i.i36.i
	br i1 undef, label %sort_LinkPrint.exit.i.i, label %bb15.i.i.i2164

bb15.i.i.i2164:		; preds = %bb14.i.i.i2163
	unreachable

sort_LinkPrint.exit.i.i:		; preds = %bb14.i.i.i2163
	unreachable

bb4.i43.i:		; preds = %bb1.i28.i
	unreachable

bb7.i2177:		; preds = %bb5.i2144
	unreachable

bb8.i2178:		; preds = %bb4.i2143
	br i1 undef, label %sort_ApproxStaticSortTheory.exit, label %bb.i5.i2185.preheader

bb.i5.i2185.preheader:		; preds = %bb8.i2178
	br label %bb.i5.i2185

bb.i5.i2185:		; preds = %bb.i5.i2185, %bb.i5.i2185.preheader
	br i1 undef, label %sort_ApproxStaticSortTheory.exit, label %bb.i5.i2185

sort_ApproxStaticSortTheory.exit:		; preds = %bb.i5.i2185, %bb8.i2178
	br label %bb25.i

bb25.i:		; preds = %sort_ApproxStaticSortTheory.exit, %bb23.i
	unreachable

bb26.i:		; preds = %bb9.i
	unreachable
}
