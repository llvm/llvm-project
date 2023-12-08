; XFAIL: *
; REQUIRES: asserts
; RUN: opt < %s -passes=newgvn -stats -disable-output 2>&1 | FileCheck %s
; rdar://7363102

; CHECK: Number of loads deleted

; GVN should be able to eliminate load %tmp22.i, because it is redundant with
; load %tmp8.i. This requires being able to prove that %tmp7.i doesn't
; alias the malloc'd value %tmp.i20.i.i, which it can do since %tmp7.i
; is derived from %tmp5.i which is computed from a load, and %tmp.i20.i.i
; is never stored and does not escape.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

%"struct.llvm::MallocAllocator" = type <{ i8 }>
%"struct.llvm::StringMap<ptr,llvm::MallocAllocator>" = type { %"struct.llvm::StringMapImpl", %"struct.llvm::MallocAllocator" }
%"struct.llvm::StringMapEntry<ptr>" = type { %"struct.llvm::StringMapEntryBase", ptr }
%"struct.llvm::StringMapEntryBase" = type { i32 }
%"struct.llvm::StringMapImpl" = type { ptr, i32, i32, i32, i32 }
%"struct.llvm::StringMapImpl::ItemBucket" = type { i32, ptr }
%"struct.llvm::StringRef" = type { ptr, i64 }

define ptr @_Z3fooRN4llvm9StringMapIPvNS_15MallocAllocatorEEEPKc(ptr %X, ptr %P) ssp {
entry:
  %tmp = alloca %"struct.llvm::StringRef", align 8
  store ptr %P, ptr %tmp, align 8
  %tmp1.i = call i64 @strlen(ptr %P) nounwind readonly
  %tmp2.i = getelementptr inbounds %"struct.llvm::StringRef", ptr %tmp, i64 0, i32 1
  store i64 %tmp1.i, ptr %tmp2.i, align 8
  %tmp1 = call ptr @_ZN4llvm9StringMapIPvNS_15MallocAllocatorEE16GetOrCreateValueERKNS_9StringRefE(ptr %X, ptr %tmp) ssp
  ret ptr %tmp1
}

declare i64 @strlen(ptr nocapture) nounwind readonly

declare noalias ptr @malloc(i64) nounwind

declare i32 @_ZN4llvm13StringMapImpl15LookupBucketForENS_9StringRefE(ptr, i64, i64)

define linkonce_odr ptr @_ZN4llvm9StringMapIPvNS_15MallocAllocatorEE16GetOrCreateValueERKNS_9StringRefE(ptr %this, ptr nocapture %Key) ssp align 2 {
entry:
  %val = load i64, ptr %Key
  %tmp = getelementptr inbounds %"struct.llvm::StringRef", ptr %Key, i64 0, i32 1
  %val2 = load i64, ptr %tmp
  %tmp2.i = getelementptr inbounds %"struct.llvm::StringMap<ptr,llvm::MallocAllocator>", ptr %this, i64 0, i32 0
  %tmp3.i = tail call i32 @_ZN4llvm13StringMapImpl15LookupBucketForENS_9StringRefE(ptr %tmp2.i, i64 %val, i64 %val2)
  %tmp4.i = getelementptr inbounds %"struct.llvm::StringMap<ptr,llvm::MallocAllocator>", ptr %this, i64 0, i32 0, i32 0
  %tmp5.i = load ptr, ptr %tmp4.i, align 8
  %tmp6.i = zext i32 %tmp3.i to i64
  %tmp7.i = getelementptr inbounds %"struct.llvm::StringMapImpl::ItemBucket", ptr %tmp5.i, i64 %tmp6.i, i32 1
  %tmp8.i = load ptr, ptr %tmp7.i, align 8
  %tmp9.i = icmp eq ptr %tmp8.i, null
  %tmp13.i = icmp eq ptr %tmp8.i, inttoptr (i64 -1 to ptr)
  %or.cond.i = or i1 %tmp9.i, %tmp13.i
  br i1 %or.cond.i, label %bb4.i, label %bb6.i

bb4.i:                                            ; preds = %entry
  %tmp41.i = inttoptr i64 %val to ptr
  %tmp4.i35.i = getelementptr inbounds i8, ptr %tmp41.i, i64 %val2
  %tmp.i.i = ptrtoint ptr %tmp4.i35.i to i64
  %tmp1.i.i = trunc i64 %tmp.i.i to i32
  %tmp3.i.i = trunc i64 %val to i32
  %tmp4.i.i = sub i32 %tmp1.i.i, %tmp3.i.i
  %tmp5.i.i = add i32 %tmp4.i.i, 17
  %tmp8.i.i = zext i32 %tmp5.i.i to i64
  %tmp.i20.i.i = tail call noalias ptr @malloc(i64 %tmp8.i.i) nounwind
  %tmp12.i.i = icmp eq ptr %tmp.i20.i.i, null
  br i1 %tmp12.i.i, label %_ZN4llvm14StringMapEntryIPvE6CreateINS_15MallocAllocatorES1_EEPS2_PKcS7_RT_T0_.exit.i, label %bb.i.i

bb.i.i:                                           ; preds = %bb4.i
  store i32 %tmp4.i.i, ptr %tmp.i20.i.i, align 4
  %tmp1.i19.i.i = getelementptr inbounds i8, ptr %tmp.i20.i.i, i64 8
  store ptr null, ptr %tmp1.i19.i.i, align 8
  br label %_ZN4llvm14StringMapEntryIPvE6CreateINS_15MallocAllocatorES1_EEPS2_PKcS7_RT_T0_.exit.i

_ZN4llvm14StringMapEntryIPvE6CreateINS_15MallocAllocatorES1_EEPS2_PKcS7_RT_T0_.exit.i: ; preds = %bb.i.i, %bb4.i
  %tmp.i18.i.i = getelementptr inbounds i8, ptr %tmp.i20.i.i, i64 16
  %tmp15.i.i = zext i32 %tmp4.i.i to i64
  tail call void @llvm.memcpy.p0.p0.i64(ptr %tmp.i18.i.i, ptr %tmp41.i, i64 %tmp15.i.i, i1 false)
  %tmp.i18.sum.i.i = add i64 %tmp15.i.i, 16
  %tmp17.i.i = getelementptr inbounds i8, ptr %tmp.i20.i.i, i64 %tmp.i18.sum.i.i
  store i8 0, ptr %tmp17.i.i, align 1
  %tmp.i.i.i = getelementptr inbounds i8, ptr %tmp.i20.i.i, i64 8
  store ptr null, ptr %tmp.i.i.i, align 8
  %tmp22.i = load ptr, ptr %tmp7.i, align 8
  %tmp24.i = icmp eq ptr %tmp22.i, inttoptr (i64 -1 to ptr)
  br i1 %tmp24.i, label %bb9.i, label %_ZN4llvm9StringMapIPvNS_15MallocAllocatorEE16GetOrCreateValueIS1_EERNS_14StringMapEntryIS1_EENS_9StringRefET_.exit

bb6.i:                                            ; preds = %entry
  ret ptr %tmp8.i

bb9.i:                                            ; preds = %_ZN4llvm14StringMapEntryIPvE6CreateINS_15MallocAllocatorES1_EEPS2_PKcS7_RT_T0_.exit.i
  %tmp25.i = getelementptr inbounds %"struct.llvm::StringMap<ptr,llvm::MallocAllocator>", ptr %this, i64 0, i32 0, i32 3
  %tmp26.i = load i32, ptr %tmp25.i, align 8
  %tmp27.i = add i32 %tmp26.i, -1
  store i32 %tmp27.i, ptr %tmp25.i, align 8
  ret ptr %tmp.i20.i.i

_ZN4llvm9StringMapIPvNS_15MallocAllocatorEE16GetOrCreateValueIS1_EERNS_14StringMapEntryIS1_EENS_9StringRefET_.exit: ; preds = %_ZN4llvm14StringMapEntryIPvE6CreateINS_15MallocAllocatorES1_EEPS2_PKcS7_RT_T0_.exit.i
  ret ptr %tmp.i20.i.i
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
