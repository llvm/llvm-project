; RUN: llc < %s -mtriple=avr | FileCheck -v -dump-input always %s

%Tstats = type <{ i8, i8 }>
@ui1 = protected local_unnamed_addr global i64 zeroinitializer, align 8
@ui2 = protected local_unnamed_addr global i64 zeroinitializer, align 8
@failed = private unnamed_addr addrspace(1) constant [12 x i8] c"test failed\00"
@stats = external protected global %Tstats, align 1

; CHECK-LABEL: main:
define protected noundef i32 @main(i32 %0, ptr nocapture readnone %1) local_unnamed_addr addrspace(1) #0 {
entry:
  store i64 94, ptr @ui1, align 8
  store i64 53, ptr @ui2, align 8
  tail call addrspace(1) void @reportFailureWithDetails(i16 ptrtoint (ptr addrspace(1) @failed to i16), i16 11, i8 2, i16 32, ptr nocapture nonnull swiftself dereferenceable(2) @stats)
  %11 = load i64, ptr @ui1, align 8
  %12 = load i64, ptr @ui2, align 8

; COM: CHECK: call __udivdi3
  %15 = udiv i64 %11, %12

; look for the buggy pattern where r30/r31 are being clobbered, corrupting the stack pointer
; CHECK-NOT: std  Z+{{[1-9]+}}, r30 
; CHECK-NOT: std  Z+{{[1-9]+}}, r31

; CHECK: call expect
  tail call addrspace(1) void @expect(i64 %15, i64 1, i16 ptrtoint (ptr addrspace(1) @failed to i16), i16 11, i8 2, i16 33)

; CHECK: ret
  ret i32 0
}

declare protected void @expect(i64, i64, i16, i16, i8, i16) local_unnamed_addr addrspace(1) #0
declare protected void @reportFailureWithDetails(i16, i16, i8, i16, ptr nocapture swiftself dereferenceable(2)) local_unnamed_addr addrspace(1) #0
attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
