; RUN: llvm-as < %s | opt -gvnpre | llvm-dis
	%"struct.ObjectArena<char>" = type { i32, i32, %"struct.ResizeArray<char*>", i8*, i8* }
	%"struct.ResizeArray<char*>" = type { i32 (...)**, %"struct.ResizeArrayRaw<char*>"* }
	%"struct.ResizeArrayRaw<char*>" = type { i8**, i8*, i32, i32, i32, float, i32 }

define void @_ZN11ObjectArenaIcED1Ev(%"struct.ObjectArena<char>"* %this) {
entry:
	br label %cond_true21

cond_true21:		; preds = %cond_true21, %entry
	%tmp215.0 = phi %"struct.ResizeArray<char*>"* [ null, %entry ], [ null, %cond_true21 ]		; <%"struct.ResizeArray<char*>"*> [#uses=1]
	%tmp2.i2 = getelementptr %"struct.ResizeArray<char*>"* %tmp215.0, i32 0, i32 1		; <%"struct.ResizeArrayRaw<char*>"**> [#uses=0]
	br label %cond_true21
}
