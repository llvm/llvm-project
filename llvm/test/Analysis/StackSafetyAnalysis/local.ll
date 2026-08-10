; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@sink = global ptr null, align 8

declare void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 %len, i1 %isvolatile)
declare void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 %isvolatile)
declare void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 %isvolatile)
declare void @llvm.memset.p0.i64(ptr %dest, i8 %val, i64 %len, i1 %isvolatile)

declare void @unknown_call(ptr %dest)
declare void @unknown_call_int(i64 %i)
declare ptr @retptr(ptr returned)

; Address leaked.
define void @LeakAddress() {
; CHECK-LABEL: @LeakAddress dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  store ptr %x, ptr @sink, align 8
  ret void
}

define void @StoreInBounds() {
; CHECK-LABEL: @StoreInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,1){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %x, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  store i8 0, ptr %x, align 1
  ret void
}

define void @StoreInBoundsCond(i64 %i) {
; CHECK-LABEL: @StoreInBoundsCond dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %x2, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %c1 = icmp sge i64 %i, 0
  %c2 = icmp slt i64 %i, 4
  br i1 %c1, label %c1.true, label %false

c1.true:
  br i1 %c2, label %c2.true, label %false

c2.true:
  %x2 = getelementptr i8, ptr %x, i64 %i
  store i8 0, ptr %x2, align 1
  br label %false

false:
  ret void
}

define void @StoreInBoundsMinMax(i64 %i) {
; CHECK-LABEL: @StoreInBoundsMinMax dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %x2, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %c1 = icmp sge i64 %i, 0
  %i1 = select i1 %c1, i64 %i, i64 0
  %c2 = icmp slt i64 %i1, 3
  %i2 = select i1 %c2, i64 %i1, i64 3
  %x2 = getelementptr i8, ptr %x, i64 %i2
  store i8 0, ptr %x2, align 1
  ret void
}

define void @StoreInBounds2() {
; CHECK-LABEL: @StoreInBounds2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 0, ptr %x, align 4
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  ret void
}

define void @StoreInBounds3() {
; CHECK-LABEL: @StoreInBounds3 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,3){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %x2, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 2
  store i8 0, ptr %x2, align 1
  ret void
}

; FIXME: ScalarEvolution does not look through ptrtoint/inttoptr.
define void @StoreInBounds4() {
; CHECK-LABEL: @StoreInBounds4 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = ptrtoint ptr %x to i64
  %x2 = add i64 %x1, 2
  %x3 = inttoptr i64 %x2 to ptr
  store i8 0, ptr %x3, align 1
  ret void
}

define void @StoreInBounds6() {
; CHECK-LABEL: @StoreInBounds6 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: x[4]: full-set, @retptr(arg0, [0,1)){{$}}
; LOCAL-NEXT: x[4]: [0,1), @retptr(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %x2, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = call ptr @retptr(ptr %x)
  store i8 0, ptr %x2, align 1
  ret void
}

define dso_local void @WriteMinMax(ptr %p) {
; CHECK-LABEL: @WriteMinMax{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %p1, align 1
; GLOBAL-NEXT: store i8 0, ptr %p2, align 1
; CHECK-EMPTY:
entry:
  %p1 = getelementptr i8, ptr %p, i64 9223372036854775805
  store i8 0, ptr %p1, align 1
  %p2 = getelementptr i8, ptr %p, i64 -9223372036854775805
  store i8 0, ptr %p2, align 1
  ret void
}

define dso_local void @WriteMax(ptr %p) {
; CHECK-LABEL: @WriteMax{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [-9223372036854775807,9223372036854775806)
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i64(ptr %p, i8 1, i64 9223372036854775806, i1 false)
; GLOBAL-NEXT: call void @llvm.memset.p0.i64(ptr %p2, i8 1, i64 9223372036854775806, i1 false)
; CHECK-EMPTY:
entry:
  call void @llvm.memset.p0.i64(ptr %p, i8 1, i64 9223372036854775806, i1 0)
  %p2 = getelementptr i8, ptr %p, i64 -9223372036854775807
  call void @llvm.memset.p0.i64(ptr %p2, i8 1, i64 9223372036854775806, i1 0)
  ret void
}

define void @StoreOutOfBounds() {
; CHECK-LABEL: @StoreOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,6){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 2
  store i32 0, ptr %x2, align 1
  ret void
}

define void @StoreOutOfBoundsCond(i64 %i) {
; CHECK-LABEL: @StoreOutOfBoundsCond dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %c1 = icmp sge i64 %i, 0
  %c2 = icmp slt i64 %i, 5
  br i1 %c1, label %c1.true, label %false

c1.true:
  br i1 %c2, label %c2.true, label %false

c2.true:
  %x2 = getelementptr i8, ptr %x, i64 %i
  store i8 0, ptr %x2, align 1
  br label %false

false:
  ret void
}

define void @StoreOutOfBoundsCond2(i64 %i) {
; CHECK-LABEL: @StoreOutOfBoundsCond2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %c2 = icmp slt i64 %i, 5
  br i1 %c2, label %c2.true, label %false

c2.true:
  %x2 = getelementptr i8, ptr %x, i64 %i
  store i8 0, ptr %x2, align 1
  br label %false

false:
  ret void
}

define void @StoreOutOfBounds2() {
; CHECK-LABEL: @StoreOutOfBounds2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: x[4]: full-set, @retptr(arg0, [2,3)){{$}}
; LOCAL-NEXT: x[4]: [2,6), @retptr(arg0, [2,3)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 2
  %x3 = call ptr @retptr(ptr %x2)
  store i32 0, ptr %x3, align 1
  ret void
}

; There is no difference in load vs store handling.
define void @LoadInBounds() {
; CHECK-LABEL: @LoadInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,1){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: %v = load i8, ptr %x, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %v = load i8, ptr %x, align 1
  ret void
}

define void @LoadOutOfBounds() {
; CHECK-LABEL: @LoadOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,6){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 2
  %v = load i32, ptr %x2, align 1
  ret void
}

; Leak through ret.
define ptr @Ret() {
; CHECK-LABEL: @Ret dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 2
  ret ptr %x2
}

declare void @Foo(ptr %p)

define void @DirectCall() {
; CHECK-LABEL: @DirectCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Foo(arg0, [2,3)){{$}}
; GLOBAL-NEXT: x[8]: full-set, @Foo(arg0, [2,3)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i16, ptr %x, i64 1
  call void @Foo(ptr %x2);
  ret void
}

; Indirect calls can not be analyzed (yet).
; FIXME: %p[]: full-set looks invalid
define void @IndirectCall(ptr %p) {
; CHECK-LABEL: @IndirectCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void %p(ptr %x);
  ret void
}

define void @NonConstantOffset(i1 zeroext %z) {
; CHECK-LABEL: @NonConstantOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; FIXME: SCEV can't look through selects.
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %x2, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %idx = select i1 %z, i64 1, i64 2
  %x2 = getelementptr i8, ptr %x, i64 %idx
  store i8 0, ptr %x2, align 1
  ret void
}

define void @NegativeOffset() {
; CHECK-LABEL: @NegativeOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [-1600000000000,-1599999999996){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x2 = getelementptr i32, ptr %x, i64 -400000000000
  store i32 0, ptr %x2, align 1
  ret void
}

define void @PossiblyNegativeOffset(i16 %z) {
; CHECK-LABEL: @PossiblyNegativeOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [-131072,131072){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x2 = getelementptr i32, ptr %x, i16 %z
  store i32 0, ptr %x2, align 1
  ret void
}

define void @NonConstantOffsetOOB(i1 zeroext %z) {
; CHECK-LABEL: @NonConstantOffsetOOB dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,6){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %idx = select i1 %z, i64 1, i64 4
  %x2 = getelementptr i8, ptr %x, i64 %idx
  store i8 0, ptr %x2, align 1
  ret void
}

define void @ArrayAlloca() {
; CHECK-LABEL: @ArrayAlloca dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [36,40){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 0, ptr %x2, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x2 = getelementptr i8, ptr %x, i64 36
  store i32 0, ptr %x2, align 1
  ret void
}

define void @ArrayAllocaOOB() {
; CHECK-LABEL: @ArrayAllocaOOB dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [37,41){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x2 = getelementptr i8, ptr %x, i64 37
  store i32 0, ptr %x2, align 1
  ret void
}

define void @DynamicAllocaUnused(i64 %size) {
; CHECK-LABEL: @DynamicAllocaUnused dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: empty-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, i64 %size, align 16
  ret void
}

; Dynamic alloca with unknown size.
define void @DynamicAlloca(i64 %size) {
; CHECK-LABEL: @DynamicAlloca dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, i64 %size, align 16
  store i32 0, ptr %x, align 1
  ret void
}

; Dynamic alloca with limited size.
; FIXME: could be proved safe. Implement.
define void @DynamicAllocaFiniteSizeRange(i1 zeroext %z) {
; CHECK-LABEL: @DynamicAllocaFiniteSizeRange dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %size = select i1 %z, i64 3, i64 5
  %x = alloca i32, i64 %size, align 16
  store i32 0, ptr %x, align 1
  ret void
}

define signext i8 @SimpleLoop() {
; CHECK-LABEL: @SimpleLoop dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[10]: [0,10){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: %load = load volatile i8, ptr %p.09, align 1
; CHECK-EMPTY:
entry:
  %x = alloca [10 x i8], align 1
  %lftr.limit = getelementptr inbounds [10 x i8], ptr %x, i64 0, i64 10
  br label %for.body

for.body:
  %sum.010 = phi i8 [ 0, %entry ], [ %add, %for.body ]
  %p.09 = phi ptr [ %x, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, ptr %p.09, i64 1
  %load = load volatile i8, ptr %p.09, align 1
  %add = add i8 %load, %sum.010
  %exitcond = icmp eq ptr %incdec.ptr, %lftr.limit
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i8 %add
}

; OOB in a loop.
define signext i8 @SimpleLoopOOB() {
; CHECK-LABEL: @SimpleLoopOOB dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[10]: [0,11){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca [10 x i8], align 1
 ; 11 iterations
  %lftr.limit = getelementptr inbounds [10 x i8], ptr %x, i64 0, i64 11
  br label %for.body

for.body:
  %sum.010 = phi i8 [ 0, %entry ], [ %add, %for.body ]
  %p.09 = phi ptr [ %x, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, ptr %p.09, i64 1
  %load = load volatile i8, ptr %p.09, align 1
  %add = add i8 %load, %sum.010
  %exitcond = icmp eq ptr %incdec.ptr, %lftr.limit
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i8 %add
}

define dso_local void @SizeCheck(i32 %sz) {
; CHECK-LABEL: @SizeCheck{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x1[128]: [0,4294967295){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x1 = alloca [128 x i8], align 16
  %cmp = icmp slt i32 %sz, 129
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @llvm.memset.p0.i32(ptr nonnull align 16 %x1, i8 0, i32 %sz, i1 false)
  br label %if.end

if.end:
  ret void
}

; FIXME: scalable allocas are considered to be of size zero, and scalable accesses to be full-range.
; This effectively disables safety analysis for scalable allocations.
define void @Scalable(ptr %p, ptr %unused, <vscale x 4 x i32> %v) {
; CHECK-LABEL: @Scalable dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT:   p[]: full-set
; CHECK-NEXT:   unused[]: empty-set
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   x[0]: [0,1){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store <vscale x 4 x i32> %v, ptr %p, align 4
; CHECK-EMPTY:
entry:
  %x = alloca <vscale x 4 x i32>, align 4
  store i8 0, ptr %x, align 1
  store <vscale x 4 x i32> %v, ptr %p, align 4
  ret void
}

%zerosize_type = type {}

define void @ZeroSize(ptr %p)  {
; CHECK-LABEL: @ZeroSize dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT:   p[]: empty-set
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   x[0]: empty-set
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store %zerosize_type undef, ptr %x, align 4
; GLOBAL-NEXT: store %zerosize_type undef, ptr undef, align 4
; GLOBAL-NEXT: load %zerosize_type, ptr %p, align
; CHECK-EMPTY:
entry:
  %x = alloca %zerosize_type, align 4
  store %zerosize_type undef, ptr %x, align 4
  store %zerosize_type undef, ptr undef, align 4
  %val = load %zerosize_type, ptr %p, align 4
  ret void
}

define void @OperandBundle() {
; CHECK-LABEL: @OperandBundle dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   a[4]: full-set
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @LeakAddress() ["unknown"(ptr %a)]
  ret void
}

define void @ByVal(ptr byval(i16) %p) {
  ; CHECK-LABEL: @ByVal dso_preemptable{{$}}
  ; CHECK-NEXT: args uses:
  ; CHECK-NEXT: allocas uses:
  ; GLOBAL-NEXT: safe accesses:
  ; CHECK-EMPTY:
entry:
  ret void
}

define void @TestByVal() {
; CHECK-LABEL: @TestByVal dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[2]: [0,2)
; CHECK-NEXT: y[8]: [0,2)
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @ByVal(ptr byval(i16) %x)
; GLOBAL-NEXT: call void @ByVal(ptr byval(i16) %y)
; CHECK-EMPTY:
entry:
  %x = alloca i16, align 4
  call void @ByVal(ptr byval(i16) %x)

  %y = alloca i64, align 4
  call void @ByVal(ptr byval(i16) %y)

  ret void
}

declare void @ByValArray(ptr byval([100000 x i64]) %p)

define void @TestByValArray() {
; CHECK-LABEL: @TestByValArray dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: z[800000]: [500000,1300000)
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %z = alloca [100000 x i64], align 4
  %z2 = getelementptr i8, ptr %z, i64 500000
  call void @ByValArray(ptr byval([100000 x i64]) %z2)
  ret void
}

define dso_local i8 @LoadMinInt64(ptr %p) {
  ; CHECK-LABEL: @LoadMinInt64{{$}}
  ; CHECK-NEXT: args uses:
  ; CHECK-NEXT: p[]: [-9223372036854775808,-9223372036854775807){{$}}
  ; CHECK-NEXT: allocas uses:
  ; GLOBAL-NEXT: safe accesses:
  ; GLOBAL-NEXT: load i8, ptr %p2, align 1
  ; CHECK-EMPTY:
  %p2 = getelementptr i8, ptr %p, i64 -9223372036854775808
  %v = load i8, ptr %p2, align 1
  ret i8 %v
}

define void @Overflow() {
; CHECK-LABEL: @Overflow dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @LoadMinInt64(arg0, [-9223372036854775808,-9223372036854775807)){{$}}
; GLOBAL-NEXT: x[1]: full-set, @LoadMinInt64(arg0, [-9223372036854775808,-9223372036854775807)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %x2 = getelementptr i8, ptr %x, i64 -9223372036854775808
  %v = call i8 @LoadMinInt64(ptr %x2)
  ret void
}

define void @DeadBlock(ptr %p) {
; CHECK-LABEL: @DeadBlock dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: empty-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[1]: empty-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 5, ptr %x
; GLOBAL-NEXT: store i64 -5, ptr %p
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  br label %end

dead:
  store i8 5, ptr %x
  store i64 -5, ptr %p
  br label %end

end:
  ret void
}

define void @LifeNotStarted() {
; CHECK-LABEL: @LifeNotStarted dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: x[1]: full-set{{$}}
; CHECK: y[1]: full-set{{$}}
; CHECK: z[1]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  %z = alloca i8, align 4

  store i8 5, ptr %x
  %n = load i8, ptr %y
  call void @llvm.memset.p0.i32(ptr nonnull %z, i8 0, i32 1, i1 false)

  call void @llvm.lifetime.start.p0(ptr %x)
  call void @llvm.lifetime.start.p0(ptr %y)
  call void @llvm.lifetime.start.p0(ptr %z)

  ret void
}

define void @LifeOK() {
; CHECK-LABEL: @LifeOK dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: x[1]: [0,1){{$}}
; CHECK: y[1]: [0,1){{$}}
; CHECK: z[1]: [0,1){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 5, ptr %x
; GLOBAL-NEXT: %n = load i8, ptr %y
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr nonnull %z, i8 0, i32 1, i1 false)
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  %z = alloca i8, align 4

  call void @llvm.lifetime.start.p0(ptr %x)
  call void @llvm.lifetime.start.p0(ptr %y)
  call void @llvm.lifetime.start.p0(ptr %z)

  store i8 5, ptr %x
  %n = load i8, ptr %y
  call void @llvm.memset.p0.i32(ptr nonnull %z, i8 0, i32 1, i1 false)

  ret void
}

define void @LifeEnded() {
; CHECK-LABEL: @LifeEnded dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: x[1]: full-set{{$}}
; CHECK: y[1]: full-set{{$}}
; CHECK: z[1]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  %z = alloca i8, align 4

  call void @llvm.lifetime.start.p0(ptr %x)
  call void @llvm.lifetime.start.p0(ptr %y)
  call void @llvm.lifetime.start.p0(ptr %z)

  call void @llvm.lifetime.end.p0(ptr %x)
  call void @llvm.lifetime.end.p0(ptr %y)
  call void @llvm.lifetime.end.p0(ptr %z)

  store i8 5, ptr %x
  %n = load i8, ptr %y
  call void @llvm.memset.p0.i32(ptr nonnull %z, i8 0, i32 1, i1 false)

  ret void
}

define void @TwoAllocasOK() {
; CHECK-LABEL: @TwoAllocasOK
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,1){{$}}
; CHECK: y[1]: [0,1){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %y, ptr %a, i32 1, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %y = alloca i8, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %y, ptr %a, i32 1, i1 false)
  ret void
}

define void @TwoAllocasOOBDest() {
; CHECK-LABEL: @TwoAllocasOOBDest
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,4){{$}}
; CHECK: y[1]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %y = alloca i8, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %y, ptr %a, i32 4, i1 false)
  ret void
}

define void @TwoAllocasOOBSource() {
; CHECK-LABEL: @TwoAllocasOOBSource
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,4){{$}}
; CHECK: y[1]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %y = alloca i8, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %y, i32 4, i1 false)
  ret void
}

define void @TwoAllocasOOBBoth() {
; CHECK-LABEL: @TwoAllocasOOBBoth
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,5){{$}}
; CHECK: y[1]: [0,5){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %y = alloca i8, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %y, ptr %a, i32 5, i1 false)
  ret void
}

define void @MixedAccesses() {
; CHECK-LABEL: @MixedAccesses
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,5){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 5, i1 false)
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
  ret void
}

define void @MixedAccesses2() {
; CHECK-LABEL: @MixedAccesses2
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,8){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: load i32, ptr %a, align 4
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %n1 = load i64, ptr %a, align 4
  %n2 = load i32, ptr %a, align 4
  ret void
}

define void @MixedAccesses3(ptr %func) {
; CHECK-LABEL: @MixedAccesses3
; CHECK-NEXT: args uses:
; CHECK-NEXT: func[]: full-set
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: load i32, ptr %a, align 4
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %n2 = load i32, ptr %a, align 4
  call void %func(ptr %a)
  ret void
}

define void @MixedAccesses4() {
; CHECK-LABEL: @MixedAccesses4
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; CHECK: a1[8]: [0,8){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: load i32, ptr %a, align 4
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %a1 = alloca ptr, align 4
  %n2 = load i32, ptr %a, align 4
  store ptr %a, ptr %a1
  ret void
}

define ptr @MixedAccesses5(i1 %x, ptr %y) {
; CHECK-LABEL: @MixedAccesses5
; CHECK-NEXT: args uses:
; CHECK: y[]: full-set
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: load i32, ptr %a, align 4
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  br i1 %x, label %tlabel, label %flabel
flabel:
  %n = load i32, ptr %a, align 4
  ret ptr %y
tlabel:
  ret ptr %a
}

define void @MixedAccesses6(ptr %arg) {
; CHECK-LABEL: @MixedAccesses6
; CHECK-NEXT: args uses:
; CHECK-NEXT: arg[]: [0,4)
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: [0,4)
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %arg, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %arg, i32 4, i1 false)
  ret void
}

define void @MixedAccesses7(i1 %cond, ptr %arg) {
; SECV doesn't support select, so we consider this non-stack-safe, even through
; it is.
;
; CHECK-LABEL: @MixedAccesses7
; CHECK-NEXT: args uses:
; CHECK-NEXT: arg[]: full-set
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  %x1 = select i1 %cond, ptr %arg, ptr %a
  call void @llvm.memcpy.p0.p0.i32(ptr %x1, ptr %arg, i32 4, i1 false)
  ret void
}

define void @NoStackAccess(ptr %arg1, ptr %arg2) {
; CHECK-LABEL: @NoStackAccess
; CHECK-NEXT: args uses:
; CHECK-NEXT: arg1[]: [0,4)
; CHECK-NEXT: arg2[]: [0,4)
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: empty-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %arg1, ptr %arg2, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %arg1, ptr %arg2, i32 4, i1 false)
  ret void
}

define void @DoubleLifetime() {
; CHECK-LABEL: @DoubleLifetime
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.lifetime.end.p0(ptr %a)
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 true)

  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
  call void @llvm.lifetime.end.p0(ptr %a)
  ret void
}

define void @DoubleLifetime2() {
; CHECK-LABEL: @DoubleLifetime2
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.lifetime.end.p0(ptr %a)
  %n = load i32, ptr %a

  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
  call void @llvm.lifetime.end.p0(ptr %a)
  ret void
}

define void @DoubleLifetime3() {
; CHECK-LABEL: @DoubleLifetime3
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.lifetime.end.p0(ptr %a)
  store i32 5, ptr %a

  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
  call void @llvm.lifetime.end.p0(ptr %a)
  ret void
}

define void @DoubleLifetime4() {
; CHECK-LABEL: @DoubleLifetime4
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: a[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.memset.p0.i32(ptr %a, i8 1, i32 4, i1 false)
  call void @llvm.lifetime.end.p0(ptr %a)
  call void @unknown_call(ptr %a)
  ret void
}

define void @Cmpxchg4Arg(ptr %p) {
; CHECK-LABEL: @Cmpxchg4Arg
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 1
; CHECK-EMPTY:
entry:
  cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 1
  ret void
}

define void @AtomicRMW4Arg(ptr %p) {
; CHECK-LABEL: @AtomicRMW4Arg
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: atomicrmw add ptr %p, i32 1 monotonic, align 1
; CHECK-EMPTY:
entry:
  atomicrmw add ptr %p, i32 1 monotonic, align 1
  ret void
}

define void @Cmpxchg4Alloca() {
; CHECK-LABEL: @Cmpxchg4Alloca
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic, align 1
  ret void
}

define void @AtomicRMW4Alloca() {
; CHECK-LABEL: @AtomicRMW4Alloca
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: atomicrmw add ptr %x, i32 1 monotonic, align 1
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  atomicrmw add ptr %x, i32 1 monotonic, align 1
  ret void
}

define void @StoreArg(ptr %p) {
; CHECK-LABEL: @StoreArg
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 1, ptr %p
; CHECK-EMPTY:
entry:
  store i32 1, ptr %p
  ret void
}

define void @NonPointer(ptr %p) {
; CHECK-LABEL: @NonPointer
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @unknown_call_int(arg0, full-set)
; GLOBAL-NEXT: p[]: full-set, @unknown_call_int(arg0, full-set)
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
  %int = ptrtoint ptr %p to i64
  call void @unknown_call_int(i64 %int)
  ret void
}

@ifunc = dso_local ifunc i64 (ptr), ptr @ifunc_resolver

define dso_local void @CallIfunc(ptr noundef %uaddr) local_unnamed_addr {
; CHECK-LABEL: @CallIfunc
; CHECK-NEXT:  args uses:
; CHECK-NEXT:    uaddr[]: full-set
entry:
  tail call i64 @ifunc(ptr noundef %uaddr)
  ret void
}

define dso_local ptr @ifunc_resolver() {
entry:
  ret ptr null
}

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
