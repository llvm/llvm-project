; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -disable-output 2>/dev/null

%struct..RefPoint = type { i32, { i32, i8, i8 } }
%struct..RefRect = type { %struct..RefPoint, %struct..RefPoint }

define i32 @BMT_CommitPartDrawObj() {
	%tmp.19111 = getelementptr %struct..RefRect, ptr null, i64 0, i32 0, i32 1, i32 2
	%tmp.20311 = getelementptr %struct..RefRect, ptr null, i64 0, i32 1, i32 1, i32 2
	ret i32 0
}
