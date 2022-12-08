; RUN: %lli -jit-kind=mcjit -remote-mcjit -O0 -mcjit-remote-process=lli-child-target%exeext %s
; XFAIL: target={{.*-windows-(gnu|msvc)}}
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

@.str = private unnamed_addr constant [6 x i8] c"data1\00", align 1
@ptr = global ptr @.str, align 4
@.str1 = private unnamed_addr constant [6 x i8] c"data2\00", align 1
@ptr2 = global ptr @.str1, align 4

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind readonly {
entry:
  %0 = load ptr, ptr @ptr, align 4
  %1 = load ptr, ptr @ptr2, align 4
  %cmp = icmp eq ptr %0, %1
  %. = zext i1 %cmp to i32
  ret i32 %.
}
