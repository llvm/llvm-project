; RUN: llc < %s
; PR2849

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.BaseBoundPtrs = type { i8*, i8* }
	%struct.HashEntry = type { %struct.BaseBoundPtrs }
	%struct.NODE = type { i8, i8, %struct.anon }
	%struct.anon = type { %struct.xlist }
	%struct.xlist = type { %struct.NODE*, %struct.NODE* }
	%struct.xvect = type { %struct.NODE** }
@hash_table_begin = external global %struct.HashEntry*

define void @obshow() {
entry:
	%tmp = load %struct.HashEntry** @hash_table_begin, align 8
	br i1 false, label %xlygetvalue.exit, label %xlygetvalue.exit

xlygetvalue.exit:
	%storemerge.in.i = phi %struct.NODE** [ null, %entry ], [ null, %entry ]
	%storemerge.i = load %struct.NODE** %storemerge.in.i
	%tmp1 = ptrtoint %struct.NODE** %storemerge.in.i to i64
	%tmp2 = lshr i64 %tmp1, 3
	%tmp3 = and i64 %tmp2, 2147483647
	%tmp4 = getelementptr %struct.HashEntry* %tmp, i64 %tmp3, i32 0, i32 1
	%tmp7 = load i8** %tmp4, align 8
	%tmp8 = getelementptr %struct.NODE* %storemerge.i, i64 0, i32 2
	%tmp9 = bitcast %struct.anon* %tmp8 to %struct.NODE***
	%tmp11 = load %struct.NODE*** %tmp9, align 8
	%tmp12 = ptrtoint %struct.NODE** %tmp11 to i64
	%tmp13 = lshr i64 %tmp12, 3
	%tmp14 = and i64 %tmp13, 2147483647
	%tmp15 = getelementptr %struct.HashEntry* %tmp, i64 %tmp14, i32 0, i32 1
	call fastcc void @xlprint(i8** %tmp4, i8* %tmp7, i8** %tmp15)
	ret void
}

declare fastcc void @xlprint(i8**, i8*, i8**)
