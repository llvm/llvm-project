; REQUIRES: asserts
; REQUIRES: systemz-registered-target
; Used to fail with: LLVM ERROR: Error while trying to spill R5D from class ADDR64Bit: Cannot scavenge register without an emergency spill slot!


; RUN: llc %s --mtriple s390x-ibm-zos -filetype obj -o %t

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; ModuleID = 'large-stack-frames.cpp'
source_filename = "large-stack-frames.cpp"
target datalayout = "E-m:l-p1:32:32-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-zos"
%struct.slice.108 = type { ptr, ptr, [8 x i64], [8 x i64], [8 x i64] }
declare void @dealloc(ptr) local_unnamed_addr #0
define internal void @foo([26 x i64] %co1, [26 x i64] %co2, [26 x i64] %co3, [26 x i64] %co4, [26 x i64] %co5, [26 x i64] %co6, [26 x i64] %co7, [26 x i64] %co8, i32 %skip_dispatch, ptr %0, i1 %1) #0 {
entry:
  %ref.tmp = alloca %struct.slice.108, align 8
  br i1 %1, label %error, label %if.end95
if.end95:
  br i1 %1, label %if.else.i1546, label %object.exit1547
if.else.i1546:
  tail call void @dealloc(ptr noundef nonnull %0)
  br label %object.exit1547
object.exit1547:
  %call96 = tail call fastcc noundef ptr @slice([26 x i64] inreg %co7, i32 noundef signext 1, ptr noundef nonnull @get_float, ptr noundef nonnull @object, i32 noundef signext 0)
  ret void
error:
  ret void
}
declare dso_local fastcc ptr @slice([26 x i64], i32, ptr, ptr, i32) unnamed_addr #0
define internal ptr @get_float(ptr %itemp, ptr %2) #0 {
entry:
  ret ptr %2
}
define internal i32 @object(ptr %itemp, ptr %obj) #0 {
entry:
  ret i32 1
}
