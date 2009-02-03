; RUN: llvm-as < %s | llc -march=x86 -mattr=-sse -mtriple=i686-apple-darwin8.8.0 | grep mov | count 7
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse -mtriple=i686-apple-darwin8.8.0 | grep mov | count 5

	%struct.ParmT = type { [25 x i8], i8, i8* }
@.str12 = internal constant [25 x i8] c"image\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"		; <[25 x i8]*> [#uses=1]

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 

define void @t(i32 %argc, i8** %argv) nounwind  {
entry:
	%parms.i = alloca [13 x %struct.ParmT]		; <[13 x %struct.ParmT]*> [#uses=1]
	%parms1.i = getelementptr [13 x %struct.ParmT]* %parms.i, i32 0, i32 0, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %parms1.i, i8* getelementptr ([25 x i8]* @.str12, i32 0, i32 0), i32 25, i32 1 ) nounwind 
	unreachable
}
