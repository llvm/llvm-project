; RUN: opt -S -analyze -stack-safety-local < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -analyze -stack-safety < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,GLOBAL
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@sink = global i8* null, align 8

declare void @llvm.memset.p0i8.i32(i8* %dest, i8 %val, i32 %len, i1 %isvolatile)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 %isvolatile)
declare void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 %isvolatile)
declare void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 %len, i1 %isvolatile)

; Address leaked.
define void @LeakAddress() {
; CHECK-LABEL: @LeakAddress dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  store i8* %x1, i8** @sink, align 8
  ret void
}

define void @StoreInBounds() {
; CHECK-LABEL: @StoreInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,1){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  store i8 0, i8* %x1, align 1
  ret void
}

define void @StoreInBounds2() {
; CHECK-LABEL: @StoreInBounds2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  store i32 0, i32* %x, align 4
  ret void
}

define void @StoreInBounds3() {
; CHECK-LABEL: @StoreInBounds3 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,3){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 2
  store i8 0, i8* %x2, align 1
  ret void
}

; FIXME: ScalarEvolution does not look through ptrtoint/inttoptr.
define void @StoreInBounds4() {
; CHECK-LABEL: @StoreInBounds4 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [-9223372036854775808,9223372036854775807){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = ptrtoint i32* %x to i64
  %x2 = add i64 %x1, 2
  %x3 = inttoptr i64 %x2 to i8*
  store i8 0, i8* %x3, align 1
  ret void
}

define dso_local void @WriteMinMax(i8* %p) {
; CHECK-LABEL: @WriteMinMax{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:
entry:
  %p1 = getelementptr i8, i8* %p, i64 9223372036854775805
  store i8 0, i8* %p1, align 1
  %p2 = getelementptr i8, i8* %p, i64 -9223372036854775805
  store i8 0, i8* %p2, align 1
  ret void
}

define dso_local void @WriteMax(i8* %p) {
; CHECK-LABEL: @WriteMax{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [-9223372036854775807,9223372036854775806)
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:
entry:
  call void @llvm.memset.p0i8.i64(i8* %p, i8 1, i64 9223372036854775806, i1 0)
  %p2 = getelementptr i8, i8* %p, i64 -9223372036854775807
  call void @llvm.memset.p0i8.i64(i8* %p2, i8 1, i64 9223372036854775806, i1 0)
  ret void
}

define void @StoreOutOfBounds() {
; CHECK-LABEL: @StoreOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,6){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 2
  %x3 = bitcast i8* %x2 to i32*
  store i32 0, i32* %x3, align 1
  ret void
}

; There is no difference in load vs store handling.
define void @LoadInBounds() {
; CHECK-LABEL: @LoadInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,1){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %v = load i8, i8* %x1, align 1
  ret void
}

define void @LoadOutOfBounds() {
; CHECK-LABEL: @LoadOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,6){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 2
  %x3 = bitcast i8* %x2 to i32*
  %v = load i32, i32* %x3, align 1
  ret void
}

; Leak through ret.
define i8* @Ret() {
; CHECK-LABEL: @Ret dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 2
  ret i8* %x2
}

declare void @Foo(i16* %p)

define void @DirectCall() {
; CHECK-LABEL: @DirectCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Foo(arg0, [2,3)){{$}}
; GLOBAL-NEXT: x[8]: full-set, @Foo(arg0, [2,3)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i16*
  %x2 = getelementptr i16, i16* %x1, i64 1
  call void @Foo(i16* %x2);
  ret void
}

; Indirect calls can not be analyzed (yet).
; FIXME: %p[]: full-set looks invalid
define void @IndirectCall(void (i8*)* %p) {
; CHECK-LABEL: @IndirectCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void %p(i8* %x1);
  ret void
}

define void @NonConstantOffset(i1 zeroext %z) {
; CHECK-LABEL: @NonConstantOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; FIXME: SCEV can't look through selects.
; CHECK-NEXT: x[4]: [-4,4){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %idx = select i1 %z, i64 1, i64 2
  %x2 = getelementptr i8, i8* %x1, i64 %idx
  store i8 0, i8* %x2, align 1
  ret void
}

define void @NegativeOffset() {
; CHECK-LABEL: @NegativeOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [-1600000000000,-1599999999996){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x2 = getelementptr i32, i32* %x, i64 -400000000000
  store i32 0, i32* %x2, align 1
  ret void
}

define void @PossiblyNegativeOffset(i16 %z) {
; CHECK-LABEL: @PossiblyNegativeOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [-131072,131072){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x2 = getelementptr i32, i32* %x, i16 %z
  store i32 0, i32* %x2, align 1
  ret void
}

define void @NonConstantOffsetOOB(i1 zeroext %z) {
; CHECK-LABEL: @NonConstantOffsetOOB dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [-8,8){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %idx = select i1 %z, i64 1, i64 4
  %x2 = getelementptr i8, i8* %x1, i64 %idx
  store i8 0, i8* %x2, align 1
  ret void
}

define void @ArrayAlloca() {
; CHECK-LABEL: @ArrayAlloca dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [36,40){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 36
  %x3 = bitcast i8* %x2 to i32*
  store i32 0, i32* %x3, align 1
  ret void
}

define void @ArrayAllocaOOB() {
; CHECK-LABEL: @ArrayAllocaOOB dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[40]: [37,41){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, i32 10, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 37
  %x3 = bitcast i8* %x2 to i32*
  store i32 0, i32* %x3, align 1
  ret void
}

define void @DynamicAllocaUnused(i64 %size) {
; CHECK-LABEL: @DynamicAllocaUnused dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: empty-set{{$}}
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
; CHECK-EMPTY:
entry:
  %x = alloca i32, i64 %size, align 16
  store i32 0, i32* %x, align 1
  ret void
}

; Dynamic alloca with limited size.
; FIXME: could be proved safe. Implement.
define void @DynamicAllocaFiniteSizeRange(i1 zeroext %z) {
; CHECK-LABEL: @DynamicAllocaFiniteSizeRange dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: [0,4){{$}}
; CHECK-EMPTY:
entry:
  %size = select i1 %z, i64 3, i64 5
  %x = alloca i32, i64 %size, align 16
  store i32 0, i32* %x, align 1
  ret void
}

define signext i8 @SimpleLoop() {
; CHECK-LABEL: @SimpleLoop dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[10]: [0,10){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca [10 x i8], align 1
  %0 = getelementptr inbounds [10 x i8], [10 x i8]* %x, i64 0, i64 0
  %lftr.limit = getelementptr inbounds [10 x i8], [10 x i8]* %x, i64 0, i64 10
  br label %for.body

for.body:
  %sum.010 = phi i8 [ 0, %entry ], [ %add, %for.body ]
  %p.09 = phi i8* [ %0, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, i8* %p.09, i64 1
  %1 = load volatile i8, i8* %p.09, align 1
  %add = add i8 %1, %sum.010
  %exitcond = icmp eq i8* %incdec.ptr, %lftr.limit
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
; CHECK-EMPTY:
entry:
  %x = alloca [10 x i8], align 1
  %0 = getelementptr inbounds [10 x i8], [10 x i8]* %x, i64 0, i64 0
 ; 11 iterations
  %lftr.limit = getelementptr inbounds [10 x i8], [10 x i8]* %x, i64 0, i64 11
  br label %for.body

for.body:
  %sum.010 = phi i8 [ 0, %entry ], [ %add, %for.body ]
  %p.09 = phi i8* [ %0, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, i8* %p.09, i64 1
  %1 = load volatile i8, i8* %p.09, align 1
  %add = add i8 %1, %sum.010
  %exitcond = icmp eq i8* %incdec.ptr, %lftr.limit
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i8 %add
}

define dso_local void @SizeCheck(i32 %sz) {
; CHECK-LABEL: @SizeCheck{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x1[128]: [0,4294967295){{$}}
; CHECK-EMPTY:
entry:
  %x1 = alloca [128 x i8], align 16
  %x1.sub = getelementptr inbounds [128 x i8], [128 x i8]* %x1, i64 0, i64 0
  %cmp = icmp slt i32 %sz, 129
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @llvm.memset.p0i8.i32(i8* nonnull align 16 %x1.sub, i8 0, i32 %sz, i1 false)
  br label %if.end

if.end:
  ret void
}

; FIXME: scalable allocas are considered to be of size zero, and scalable accesses to be full-range.
; This effectively disables safety analysis for scalable allocations.
define void @Scalable(<vscale x 4 x i32>* %p, <vscale x 4 x i32>* %unused, <vscale x 4 x i32> %v) {
; CHECK-LABEL: @Scalable dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT:   p[]: full-set
; CHECK-NEXT:   unused[]: empty-set
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   x[0]: [0,1){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca <vscale x 4 x i32>, align 4
  %x1 = bitcast <vscale x 4 x i32>* %x to i8*
  store i8 0, i8* %x1, align 1
  store <vscale x 4 x i32> %v, <vscale x 4 x i32>* %p, align 4
  ret void
}

%zerosize_type = type {}

define void @ZeroSize(%zerosize_type *%p)  {
; CHECK-LABEL: @ZeroSize dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT:   p[]: empty-set
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   x[0]: empty-set
; CHECK-EMPTY:
entry:
  %x = alloca %zerosize_type, align 4
  store %zerosize_type undef, %zerosize_type* %x, align 4
  store %zerosize_type undef, %zerosize_type* undef, align 4
  %val = load %zerosize_type, %zerosize_type* %p, align 4
  ret void
}

define void @OperandBundle() {
; CHECK-LABEL: @OperandBundle dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   a[4]: full-set
; CHECK-EMPTY:
entry:
  %a = alloca i32, align 4
  call void @LeakAddress() ["unknown"(i32* %a)]
  ret void
}

define void @ByVal(i16* byval(i16) %p) {
  ; CHECK-LABEL: @ByVal dso_preemptable{{$}}
  ; CHECK-NEXT: args uses:
  ; CHECK-NEXT: allocas uses:
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
; CHECK-EMPTY:
entry:
  %x = alloca i16, align 4
  call void @ByVal(i16* byval(i16) %x)

  %y = alloca i64, align 4
  %y1 = bitcast i64* %y to i16*
  call void @ByVal(i16* byval(i16) %y1)

  ret void
}

declare void @ByValArray([100000 x i64]* byval([100000 x i64]) %p)

define void @TestByValArray() {
; CHECK-LABEL: @TestByValArray dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: z[800000]: [500000,1300000)
; CHECK-EMPTY:
entry:
  %z = alloca [100000 x i64], align 4
  %z1 = bitcast [100000 x i64]* %z to i8*
  %z2 = getelementptr i8, i8* %z1, i64 500000
  %z3 = bitcast i8* %z2 to [100000 x i64]*
  call void @ByValArray([100000 x i64]* byval([100000 x i64]) %z3)
  ret void
}

define dso_local i8 @LoadMinInt64(i8* %p) {
  ; CHECK-LABEL: @LoadMinInt64{{$}}
  ; CHECK-NEXT: args uses:
  ; CHECK-NEXT: p[]: [-9223372036854775808,-9223372036854775807){{$}}
  ; CHECK-NEXT: allocas uses:
  ; CHECK-EMPTY:
  %p2 = getelementptr i8, i8* %p, i64 -9223372036854775808
  %v = load i8, i8* %p2, align 1
  ret i8 %v
}

define void @Overflow() {
; CHECK-LABEL: @Overflow dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @LoadMinInt64(arg0, [-9223372036854775808,-9223372036854775807)){{$}}
; GLOBAL-NEXT: x[1]: full-set, @LoadMinInt64(arg0, [-9223372036854775808,-9223372036854775807)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %x2 = getelementptr i8, i8* %x, i64 -9223372036854775808
  %v = call i8 @LoadMinInt64(i8* %x2)
  ret void
}

define void @DeadBlock(i64* %p) {
; CHECK-LABEL: @DeadBlock dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: empty-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[1]: empty-set{{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  br label %end

dead:
  store i8 5, i8* %x
  store i64 -5, i64* %p
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
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  %z = alloca i8, align 4

  store i8 5, i8* %x
  %n = load i8, i8* %y
  call void @llvm.memset.p0i8.i32(i8* nonnull %z, i8 0, i32 1, i1 false)

  call void @llvm.lifetime.start.p0i8(i64 1, i8* %x)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %y)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %z)

  ret void
}

define void @LifeOK() {
; CHECK-LABEL: @LifeOK dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: x[1]: [0,1){{$}}
; CHECK: y[1]: [0,1){{$}}
; CHECK: z[1]: [0,1){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  %z = alloca i8, align 4

  call void @llvm.lifetime.start.p0i8(i64 1, i8* %x)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %y)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %z)

  store i8 5, i8* %x
  %n = load i8, i8* %y
  call void @llvm.memset.p0i8.i32(i8* nonnull %z, i8 0, i32 1, i1 false)

  ret void
}

define void @LifeEnded() {
; CHECK-LABEL: @LifeEnded dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK: x[1]: full-set{{$}}
; CHECK: y[1]: full-set{{$}}
; CHECK: z[1]: full-set{{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  %z = alloca i8, align 4

  call void @llvm.lifetime.start.p0i8(i64 1, i8* %x)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %y)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %z)

  call void @llvm.lifetime.end.p0i8(i64 1, i8* %x)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %y)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %z)

  store i8 5, i8* %x
  %n = load i8, i8* %y
  call void @llvm.memset.p0i8.i32(i8* nonnull %z, i8 0, i32 1, i1 false)

  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
