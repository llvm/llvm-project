; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr84478.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr84478.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@poolfilearr = internal unnamed_addr constant [1733 x ptr] [ptr @.str, ptr @.str.1, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.3, ptr @.str.2, ptr @.str.2, ptr @.str.4, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr @.str.2, ptr null], align 8
@strpool = dso_local local_unnamed_addr global ptr null, align 8
@poolptr = dso_local local_unnamed_addr global i64 0, align 8
@.str = private unnamed_addr constant [3 x i8] c"mu\00", align 1
@.str.1 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"xx\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"xxx\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i64 @makestring() local_unnamed_addr #0 {
  ret i64 1
}

; Function Attrs: nofree norecurse nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local range(i64 0, 2) i64 @loadpoolstrings(i64 noundef %0) local_unnamed_addr #1 {
  br label %7

2:                                                ; preds = %18, %16
  %3 = add nuw nsw i64 %8, 1
  %4 = getelementptr inbounds nuw ptr, ptr @poolfilearr, i64 %8
  %5 = load ptr, ptr %4, align 8, !tbaa !6
  %6 = icmp eq ptr %5, null
  br i1 %6, label %29, label %7, !llvm.loop !11

7:                                                ; preds = %1, %2
  %8 = phi i64 [ 1, %1 ], [ %3, %2 ]
  %9 = phi ptr [ @.str, %1 ], [ %5, %2 ]
  %10 = phi i32 [ 0, %1 ], [ %13, %2 ]
  %11 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %9) #7
  %12 = trunc i64 %11 to i32
  %13 = add nsw i32 %10, %12
  %14 = sext i32 %13 to i64
  %15 = icmp sgt i64 %0, %14
  br i1 %15, label %16, label %29

16:                                               ; preds = %7
  %17 = icmp sgt i32 %12, 0
  br i1 %17, label %18, label %2

18:                                               ; preds = %16, %18
  %19 = phi i32 [ %21, %18 ], [ %12, %16 ]
  %20 = phi ptr [ %22, %18 ], [ %9, %16 ]
  %21 = add nsw i32 %19, -1
  %22 = getelementptr inbounds nuw i8, ptr %20, i64 1
  %23 = load i8, ptr %20, align 1, !tbaa !13
  %24 = load ptr, ptr @strpool, align 8, !tbaa !6
  %25 = load i64, ptr @poolptr, align 8, !tbaa !14
  %26 = add nsw i64 %25, 1
  store i64 %26, ptr @poolptr, align 8, !tbaa !14
  %27 = getelementptr inbounds i8, ptr %24, i64 %25
  store i8 %23, ptr %27, align 1, !tbaa !13
  %28 = icmp samesign ugt i32 %19, 1
  br i1 %28, label %18, label %2, !llvm.loop !16

29:                                               ; preds = %7, %2
  %30 = phi i64 [ 1, %2 ], [ 0, %7 ]
  ret i64 %30
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca i32, align 4
  %2 = tail call dereferenceable_or_null(4000) ptr @malloc(i64 noundef 4000) #7
  store ptr %2, ptr @strpool, align 8, !tbaa !6
  %3 = icmp eq ptr %2, null
  br i1 %3, label %34, label %4

4:                                                ; preds = %0
  tail call void asm sideeffect "", "~{memory}"() #7, !srcloc !17
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  br label %10

5:                                                ; preds = %20, %18
  %6 = add nuw nsw i64 %11, 1
  %7 = getelementptr inbounds nuw ptr, ptr @poolfilearr, i64 %11
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  %9 = icmp eq ptr %8, null
  br i1 %9, label %31, label %10, !llvm.loop !11

10:                                               ; preds = %5, %4
  %11 = phi i64 [ 1, %4 ], [ %6, %5 ]
  %12 = phi ptr [ @.str, %4 ], [ %8, %5 ]
  %13 = phi i32 [ 0, %4 ], [ %16, %5 ]
  %14 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %12) #7
  %15 = trunc i64 %14 to i32
  %16 = add nsw i32 %13, %15
  %17 = icmp slt i32 %16, 4000
  br i1 %17, label %18, label %31

18:                                               ; preds = %10
  %19 = icmp sgt i32 %15, 0
  br i1 %19, label %20, label %5

20:                                               ; preds = %18, %20
  %21 = phi i32 [ %23, %20 ], [ %15, %18 ]
  %22 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %23 = add nsw i32 %21, -1
  %24 = getelementptr inbounds nuw i8, ptr %22, i64 1
  %25 = load i8, ptr %22, align 1, !tbaa !13
  %26 = load ptr, ptr @strpool, align 8, !tbaa !6
  %27 = load i64, ptr @poolptr, align 8, !tbaa !14
  %28 = add nsw i64 %27, 1
  store i64 %28, ptr @poolptr, align 8, !tbaa !14
  %29 = getelementptr inbounds i8, ptr %26, i64 %27
  store i8 %25, ptr %29, align 1, !tbaa !13
  %30 = icmp samesign ugt i32 %21, 1
  br i1 %30, label %20, label %5, !llvm.loop !16

31:                                               ; preds = %5, %10
  %32 = phi i32 [ 1, %5 ], [ 0, %10 ]
  store volatile i32 %32, ptr %1, align 4, !tbaa !18
  %33 = load ptr, ptr @strpool, align 8, !tbaa !6
  tail call void @free(ptr noundef %33) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  br label %34

34:                                               ; preds = %0, %31
  ret i32 0
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!9, !9, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"long", !9, i64 0}
!16 = distinct !{!16, !12}
!17 = !{i64 776}
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !9, i64 0}
