; RUN: %lli -jit-kind=mcjit -O0 -relocation-model=pic -code-model=small %s
; RUN: %lli -lljit-platform=Inactive -O0 -relocation-model=pic -code-model=small %s
; XFAIL: target={{(mips|mipsel)-.*}}, target={{(aarch64|arm).*}}, target={{(i686|i386).*}}

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

