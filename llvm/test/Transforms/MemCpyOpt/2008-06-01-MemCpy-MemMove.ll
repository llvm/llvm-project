; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | grep {call.*memmove.*arg1.*}
; PR2401

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.Info = type <{ i32, i32, i8*, i8*, i8*, [32 x i8*], i32, [32 x i32], i32, i32, i32, [32 x i32] }>
	%struct.S98 = type <{ [31 x double] }>
	%struct._IO_FILE = type <{ i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }>
	%struct._IO_marker = type <{ %struct._IO_marker*, %struct._IO_FILE*, i32 }>
	%struct.anon = type <{  }>
	%union.anon = type {  }
@info = common global %struct.Info zeroinitializer, align 4		; <%struct.Info*> [#uses=13]
@fails = common global i32 0, align 4		; <i32*> [#uses=37]
@s98 = common global %struct.S98 zeroinitializer, align 4		; <%struct.S98*> [#uses=2]
@a98 = common global [5 x %struct.S98] zeroinitializer, align 4		; <[5 x %struct.S98]*> [#uses=5]
@stdout = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=1]

declare void @llvm.memmove.i32(i8*, i8*, i32, i32) nounwind 

define void @test98() nounwind  {
entry:
	%arg = alloca %struct.S98, align 8		; <%struct.S98*> [#uses=2]
	%tmp13 = alloca %struct.S98		; <%struct.S98*> [#uses=2]
	%tmp14 = alloca %struct.S98		; <%struct.S98*> [#uses=2]
	%tmp15 = alloca %struct.S98		; <%struct.S98*> [#uses=2]
	%tmp17 = alloca %struct.S98		; <%struct.S98*> [#uses=2]
	%tmp21 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	%tmp23 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	%tmp25 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	%tmp27 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	%tmp29 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	%tmp31 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	%tmp33 = alloca %struct.S98		; <%struct.S98*> [#uses=0]
	call void @llvm.memset.i32( i8* bitcast (%struct.S98* @s98 to i8*), i8 0, i32 248, i32 4 )
	call void @llvm.memset.i32( i8* bitcast ([5 x %struct.S98]* @a98 to i8*), i8 0, i32 1240, i32 4 )
	call void @llvm.memset.i32( i8* bitcast (%struct.Info* @info to i8*), i8 0, i32 420, i32 4 )
	store i8* bitcast (%struct.S98* @s98 to i8*), i8** getelementptr (%struct.Info* @info, i32 0, i32 2)
	store i8* bitcast ([5 x %struct.S98]* @a98 to i8*), i8** getelementptr (%struct.Info* @info, i32 0, i32 3)
	store i8* bitcast (%struct.S98* getelementptr ([5 x %struct.S98]* @a98, i32 0, i32 3) to i8*), i8** getelementptr (%struct.Info* @info, i32 0, i32 4)
	store i32 248, i32* getelementptr (%struct.Info* @info, i32 0, i32 6)
	store i32 4, i32* getelementptr (%struct.Info* @info, i32 0, i32 8)
	store i32 4, i32* getelementptr (%struct.Info* @info, i32 0, i32 9)
	store i32 4, i32* getelementptr (%struct.Info* @info, i32 0, i32 10)
	%tmp = load i32* getelementptr (%struct.Info* @info, i32 0, i32 8)		; <i32> [#uses=1]
	%sub = add i32 %tmp, -1		; <i32> [#uses=1]
	%and = and i32 %sub, ptrtoint (%struct.S98* getelementptr ([5 x %struct.S98]* @a98, i32 0, i32 3) to i32)		; <i32> [#uses=1]
	%tobool = icmp eq i32 %and, 0		; <i1> [#uses=1]
	br i1 %tobool, label %ifend, label %ifthen

ifthen:		; preds = %entry
	%tmp3 = load i32* @fails		; <i32> [#uses=1]
	%inc = add i32 %tmp3, 1		; <i32> [#uses=1]
	store i32 %inc, i32* @fails
	br label %ifend

ifend:		; preds = %ifthen, %entry
	store i8* bitcast (double* getelementptr (%struct.S98* @s98, i32 0, i32 0, i32 18) to i8*), i8** getelementptr (%struct.Info* @info, i32 0, i32 5, i32 0)
	store i32 8, i32* getelementptr (%struct.Info* @info, i32 0, i32 7, i32 0)
	store i32 4, i32* getelementptr (%struct.Info* @info, i32 0, i32 11, i32 0)
	store double 0xC1075E4620000000, double* getelementptr (%struct.S98* @s98, i32 0, i32 0, i32 18)
	store double 0x410CD219E0000000, double* getelementptr ([5 x %struct.S98]* @a98, i32 0, i32 2, i32 0, i32 18)
	store i32 1, i32* getelementptr (%struct.Info* @info, i32 0, i32 0)
	store i32 0, i32* getelementptr (%struct.Info* @info, i32 0, i32 1)
	%tmp16 = bitcast %struct.S98* %tmp15 to i8*		; <i8*> [#uses=1]
	call void @llvm.memmove.i32( i8* %tmp16, i8* bitcast (%struct.S98* @s98 to i8*), i32 248, i32 4 )
	%tmp18 = bitcast %struct.S98* %tmp17 to i8*		; <i8*> [#uses=1]
	call void @llvm.memmove.i32( i8* %tmp18, i8* bitcast (%struct.S98* getelementptr ([5 x %struct.S98]* @a98, i32 0, i32 2) to i8*), i32 248, i32 4 )
	call void @check98( %struct.S98* sret  %tmp14, %struct.S98* byval  %tmp15, %struct.S98* getelementptr ([5 x %struct.S98]* @a98, i32 0, i32 1), %struct.S98* byval  %tmp17 )
	%tmp19 = bitcast %struct.S98* %tmp13 to i8*		; <i8*> [#uses=1]
	%tmp20 = bitcast %struct.S98* %tmp14 to i8*		; <i8*> [#uses=1]
	call void @llvm.memmove.i32( i8* %tmp19, i8* %tmp20, i32 248, i32 8 )
	%tmp1 = bitcast %struct.S98* %arg to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.S98* %tmp13 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64( i8* %tmp1, i8* %tmp2, i64 248, i32 8 )
	%arrayidx.i = getelementptr %struct.S98* %arg, i32 0, i32 0, i32 18		; <double*> [#uses=1]
	%tmp1.i = load double* %arrayidx.i, align 8		; <double> [#uses=1]
	%tmp2.i = load double* getelementptr (%struct.S98* @s98, i32 0, i32 0, i32 18)		; <double> [#uses=1]
	%cmp.i = fcmp une double %tmp1.i, %tmp2.i		; <i1> [#uses=1]
	br i1 %cmp.i, label %ifthen.i, label %checkx98.exit

ifthen.i:		; preds = %ifend
	%tmp3.i = load i32* @fails		; <i32> [#uses=1]
	%inc.i = add i32 %tmp3.i, 1		; <i32> [#uses=1]
	store i32 %inc.i, i32* @fails
	br label %checkx98.exit

checkx98.exit:		; preds = %ifthen.i, %ifend
	ret void
}

declare void @check98(%struct.S98* sret  %agg.result, %struct.S98* byval  %arg0, %struct.S98* %arg1, %struct.S98* byval  %arg2) nounwind

declare void @llvm.va_start(i8*) nounwind 

declare void @llvm.va_end(i8*) nounwind 

declare i32 @main() noreturn

declare i32 @fflush(%struct._IO_FILE*)

declare void @abort() noreturn nounwind 

declare void @exit(i32) noreturn nounwind 

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind 

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32) nounwind 
