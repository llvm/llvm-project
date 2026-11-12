; RUN: llc -relocation-model=pic < %s | grep -E ': ?\s*$' | sort | uniq -d | count 0
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This function produces a duplicate LPC label unless special care is taken when duplicating a t2LDRpci_pic instruction.

%struct.PlatformMutex = type { i32, [40 x i8] }
%struct.SpinLock = type { %struct.PlatformMutex }
%"struct.WTF::TCMalloc_ThreadCache" = type { i32, ptr, i8, [68 x %"struct.WTF::TCMalloc_ThreadCache_FreeList"], i32, i32, ptr, ptr }
%"struct.WTF::TCMalloc_ThreadCache_FreeList" = type { ptr, i16, i16 }
%struct.__darwin_pthread_handler_rec = type { ptr, ptr, ptr }
%struct._opaque_pthread_t = type { i32, ptr, [596 x i8] }

@_ZN3WTFL8heap_keyE = internal global i32 0       ; <ptr> [#uses=1]
@_ZN3WTFL10tsd_initedE.b = internal global i1 false ; <ptr> [#uses=2]
@_ZN3WTFL13pageheap_lockE = internal global %struct.SpinLock { %struct.PlatformMutex { i32 850045863, [40 x i8] zeroinitializer } } ; <ptr> [#uses=1]
@_ZN3WTFL12thread_heapsE = internal global ptr null ; <ptr> [#uses=1]
@llvm.used = appending global [1 x ptr] [ptr @_ZN3WTF20TCMalloc_ThreadCache22CreateCacheIfNecessaryEv], section "llvm.metadata" ; <ptr> [#uses=0]

define ptr @_ZN3WTF20TCMalloc_ThreadCache22CreateCacheIfNecessaryEv() nounwind {
entry:
  %0 = tail call  i32 @pthread_mutex_lock(ptr @_ZN3WTFL13pageheap_lockE) nounwind
  %.b24 = load i1, ptr @_ZN3WTFL10tsd_initedE.b, align 4 ; <i1> [#uses=1]
  br i1 %.b24, label %bb5, label %bb6

bb5:                                              ; preds = %entry
  %1 = tail call  ptr @pthread_self() nounwind
  br label %bb6

bb6:                                              ; preds = %bb5, %entry
  %me.0 = phi ptr [ %1, %bb5 ], [ null, %entry ] ; <ptr> [#uses=2]
  br label %bb11

bb7:                                              ; preds = %bb11
  %2 = getelementptr inbounds %"struct.WTF::TCMalloc_ThreadCache", ptr %h.0, i32 0, i32 1
  %3 = load ptr, ptr %2, align 4
  %4 = tail call  i32 @pthread_equal(ptr %3, ptr %me.0) nounwind
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %bb10, label %bb14

bb10:                                             ; preds = %bb7
  %6 = getelementptr inbounds %"struct.WTF::TCMalloc_ThreadCache", ptr %h.0, i32 0, i32 6
  br label %bb11

bb11:                                             ; preds = %bb10, %bb6
  %h.0.in = phi ptr [ @_ZN3WTFL12thread_heapsE, %bb6 ], [ %6, %bb10 ] ; <ptr> [#uses=1]
  %h.0 = load ptr, ptr %h.0.in, align 4 ; <ptr> [#uses=4]
  %7 = icmp eq ptr %h.0, null
  br i1 %7, label %bb13, label %bb7

bb13:                                             ; preds = %bb11
  %8 = tail call  ptr @_ZN3WTF20TCMalloc_ThreadCache7NewHeapEP17_opaque_pthread_t(ptr %me.0) nounwind
  br label %bb14

bb14:                                             ; preds = %bb13, %bb7
  %heap.1 = phi ptr [ %8, %bb13 ], [ %h.0, %bb7 ] ; <ptr> [#uses=4]
  %9 = tail call  i32 @pthread_mutex_unlock(ptr @_ZN3WTFL13pageheap_lockE) nounwind
  %10 = getelementptr inbounds %"struct.WTF::TCMalloc_ThreadCache", ptr %heap.1, i32 0, i32 2
  %11 = load i8, ptr %10, align 4
  %toBool15not = icmp eq i8 %11, 0                ; <i1> [#uses=1]
  br i1 %toBool15not, label %bb19, label %bb22

bb19:                                             ; preds = %bb14
  %.b = load i1, ptr @_ZN3WTFL10tsd_initedE.b, align 4 ; <i1> [#uses=1]
  br i1 %.b, label %bb21, label %bb22

bb21:                                             ; preds = %bb19
  store i8 1, ptr %10, align 4
  %12 = load i32, ptr @_ZN3WTFL8heap_keyE, align 4
  %13 = tail call  i32 @pthread_setspecific(i32 %12, ptr %heap.1) nounwind
  ret ptr %heap.1

bb22:                                             ; preds = %bb19, %bb14
  ret ptr %heap.1
}

declare i32 @pthread_mutex_lock(ptr)

declare i32 @pthread_mutex_unlock(ptr)

declare hidden ptr @_ZN3WTF20TCMalloc_ThreadCache7NewHeapEP17_opaque_pthread_t(ptr) nounwind

declare i32 @pthread_setspecific(i32, ptr)

declare ptr @pthread_self()

declare i32 @pthread_equal(ptr, ptr)

