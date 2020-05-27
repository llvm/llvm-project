; RUN: opt -S -analyze -stack-safety-local < %s | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -analyze -stack-safety < %s | FileCheck %s --check-prefixes=CHECK,GLOBAL
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@sink = global i8* null, align 8

declare void @llvm.memset.p0i8.i32(i8* %dest, i8 %val, i32 %len, i1 %isvolatile)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 %isvolatile)
declare void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 %isvolatile)

; Address leaked.
define void @LeakAddress() {
; CHECK-LABEL: @LeakAddress dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NEXT: x[4]: [2,-1){{$}}
; CHECK-NOT: ]:
entry:
  %x = alloca i32, align 4
  %x1 = ptrtoint i32* %x to i64
  %x2 = add i64 %x1, 2
  %x3 = inttoptr i64 %x2 to i8*
  store i8 0, i8* %x3, align 1
  ret void
}

define void @StoreOutOfBounds() {
; CHECK-LABEL: @StoreOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [2,6){{$}}
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void %p(i8* %x1);
  ret void
}

define void @NonConstantOffset(i1 zeroext %z) {
; CHECK-LABEL: @NonConstantOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: z[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; FIXME: SCEV can't look through selects.
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-NOT: ]:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %idx = select i1 %z, i64 1, i64 2
  %x2 = getelementptr i8, i8* %x1, i64 %idx
  store i8 0, i8* %x2, align 1
  ret void
}

define void @NonConstantOffsetOOB(i1 zeroext %z) {
; CHECK-LABEL: @NonConstantOffsetOOB dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: z[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,6){{$}}
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NEXT: size[]: empty-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: empty-set{{$}}
; CHECK-NOT: ]:
entry:
  %x = alloca i32, i64 %size, align 16
  ret void
}

; Dynamic alloca with unknown size.
define void @DynamicAlloca(i64 %size) {
; CHECK-LABEL: @DynamicAlloca dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: size[]: [0,-12){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: [0,4){{$}}
; CHECK-NOT: ]:
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
; CHECK-NEXT: z[]: [0,-12){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[0]: [0,4){{$}}
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
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

; FIXME: we don't understand that %sz in the memset call is limited to 128 by the preceding check.
define dso_local void @SizeCheck(i32 %sz) {
; CHECK-LABEL: @SizeCheck{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: sz[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x1[128]: full-set{{$}}
; CHECK-NOT: ]:
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
; CHECK-NEXT:   v[]: full-set
; CHECK-NEXT: allocas uses:
; CHECK-NEXT:   x[0]: [0,1){{$}}
; CHECK-NOT: ]:
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
; CHECK-NOT: ]:
entry:
  %x = alloca %zerosize_type, align 4
  store %zerosize_type undef, %zerosize_type* %x, align 4
  store %zerosize_type undef, %zerosize_type* undef, align 4
  %val = load %zerosize_type, %zerosize_type* %p, align 4
  ret void
}
