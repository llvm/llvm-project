; RUN: %lli -remote-mcjit -mcjit-remote-process=lli-child-target%exeext \
; RUN:   -O0 -relocation-model=pic -code-model=small %s
; XFAIL: mips-, mipsel-, aarch64, arm, i686, i386, mingw32, win32
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

@.str = private unnamed_addr constant [6 x i8] c"data1\00", align 1
@ptr = global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0), align 4
@.str1 = private unnamed_addr constant [6 x i8] c"data2\00", align 1
@ptr2 = global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0), align 4

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind readonly {
entry:
  %0 = load i8*, i8** @ptr, align 4
  %1 = load i8*, i8** @ptr2, align 4
  %cmp = icmp eq i8* %0, %1
  %. = zext i1 %cmp to i32
  ret i32 %.
}

