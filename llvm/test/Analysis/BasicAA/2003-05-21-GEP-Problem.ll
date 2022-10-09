; RUN: opt < %s -aa-pipeline=basic-aa -licm -disable-output
	%struct..apr_array_header_t = type { i32*, i32, i32, i32, i8* }
	%struct..apr_table_t = type { %struct..apr_array_header_t, i32, [32 x i32], [32 x i32] }

define void @table_reindex(%struct..apr_table_t* %t.1) {		; No predecessors!
	br label %loopentry

loopentry:		; preds = %0, %no_exit
	%tmp.101 = getelementptr %struct..apr_table_t, %struct..apr_table_t* %t.1, i64 0, i32 0, i32 2
	%tmp.11 = load i32, i32* %tmp.101		; <i32> [#uses=0]
	br i1 false, label %no_exit, label %UnifiedExitNode

no_exit:		; preds = %loopentry
	%tmp.25 = sext i32 0 to i64		; <i64> [#uses=1]
	%tmp.261 = getelementptr %struct..apr_table_t, %struct..apr_table_t* %t.1, i64 0, i32 3, i64 %tmp.25		; <i32*> [#uses=1]
	store i32 0, i32* %tmp.261
	br label %loopentry

UnifiedExitNode:		; preds = %loopentry
	ret void
}
