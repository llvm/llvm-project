; ModuleID = 'bytes_computed.c'
source_filename = "bytes_computed.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"

@total_computed = local_unnamed_addr global i32 0, align 4

define void @__bytes_computed_pre_numeric(i32 noundef %size, i32 noundef %id) local_unnamed_addr {
entry:
  %0 = load i32, ptr @total_computed, align 4
  %add = add nsw i32 %0, %size
  store i32 %add, ptr @total_computed, align 4
  ret void
}
