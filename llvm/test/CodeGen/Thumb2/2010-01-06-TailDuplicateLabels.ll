; RUN: llc -relocation-model=pic < %s | grep ":$" | sort | uniq -d | count 0
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This function produces a duplicate LPC label unless special care is taken when duplicating a t2LDRpci_pic instruction.

%struct.PlatformMutex = type { i32, [40 x i8] }
%struct.SpinLock = type { %struct.PlatformMutex }
%"struct.WTF::TCMalloc_ThreadCache" = type { i32, %struct._opaque_pthread_t*, i8, [68 x %"struct.WTF::TCMalloc_ThreadCache_FreeList"], i32, i32, %"struct.WTF::TCMalloc_ThreadCache"*, %"struct.WTF::TCMalloc_ThreadCache"* }
%"struct.WTF::TCMalloc_ThreadCache_FreeList" = type { i8*, i16, i16 }
%struct.__darwin_pthread_handler_rec = type { void (i8*)*, i8*, %struct.__darwin_pthread_handler_rec* }
%struct._opaque_pthread_t = type { i32, %struct.__darwin_pthread_handler_rec*, [596 x i8] }

@_ZN3WTFL8heap_keyE = internal global i32 0       ; <i32*> [#uses=1]
@_ZN3WTFL10tsd_initedE.b = internal global i1 false ; <i1*> [#uses=2]
@_ZN3WTFL13pageheap_lockE = internal global %struct.SpinLock { %struct.PlatformMutex { i32 850045863, [40 x i8] zeroinitializer } } ; <%struct.SpinLock*> [#uses=1]
@_ZN3WTFL12thread_heapsE = internal global %"struct.WTF::TCMalloc_ThreadCache"* null ; <%"struct.WTF::TCMalloc_ThreadCache"**> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (%"struct.WTF::TCMalloc_ThreadCache"* ()* @_ZN3WTF20TCMalloc_ThreadCache22CreateCacheIfNecessaryEv to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define %"struct.WTF::TCMalloc_ThreadCache"* @_ZN3WTF20TCMalloc_ThreadCache22CreateCacheIfNecessaryEv() nounwind {
entry:
  %0 = tail call  i32 @pthread_mutex_lock(%struct.PlatformMutex* getelementptr inbounds (%struct.SpinLock* @_ZN3WTFL13pageheap_lockE, i32 0, i32 0)) nounwind
  %.b24 = load i1* @_ZN3WTFL10tsd_initedE.b, align 4 ; <i1> [#uses=1]
  br i1 %.b24, label %bb5, label %bb6

bb5:                                              ; preds = %entry
  %1 = tail call  %struct._opaque_pthread_t* @pthread_self() nounwind
  br label %bb6

bb6:                                              ; preds = %bb5, %entry
  %me.0 = phi %struct._opaque_pthread_t* [ %1, %bb5 ], [ null, %entry ] ; <%struct._opaque_pthread_t*> [#uses=2]
  br label %bb11

bb7:                                              ; preds = %bb11
  %2 = getelementptr inbounds %"struct.WTF::TCMalloc_ThreadCache"* %h.0, i32 0, i32 1
  %3 = load %struct._opaque_pthread_t** %2, align 4
  %4 = tail call  i32 @pthread_equal(%struct._opaque_pthread_t* %3, %struct._opaque_pthread_t* %me.0) nounwind
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %bb10, label %bb14

bb10:                                             ; preds = %bb7
  %6 = getelementptr inbounds %"struct.WTF::TCMalloc_ThreadCache"* %h.0, i32 0, i32 6
  br label %bb11

bb11:                                             ; preds = %bb10, %bb6
  %h.0.in = phi %"struct.WTF::TCMalloc_ThreadCache"** [ @_ZN3WTFL12thread_heapsE, %bb6 ], [ %6, %bb10 ] ; <%"struct.WTF::TCMalloc_ThreadCache"**> [#uses=1]
  %h.0 = load %"struct.WTF::TCMalloc_ThreadCache"** %h.0.in, align 4 ; <%"struct.WTF::TCMalloc_ThreadCache"*> [#uses=4]
  %7 = icmp eq %"struct.WTF::TCMalloc_ThreadCache"* %h.0, null
  br i1 %7, label %bb13, label %bb7

bb13:                                             ; preds = %bb11
  %8 = tail call  %"struct.WTF::TCMalloc_ThreadCache"* @_ZN3WTF20TCMalloc_ThreadCache7NewHeapEP17_opaque_pthread_t(%struct._opaque_pthread_t* %me.0) nounwind
  br label %bb14

bb14:                                             ; preds = %bb13, %bb7
  %heap.1 = phi %"struct.WTF::TCMalloc_ThreadCache"* [ %8, %bb13 ], [ %h.0, %bb7 ] ; <%"struct.WTF::TCMalloc_ThreadCache"*> [#uses=4]
  %9 = tail call  i32 @pthread_mutex_unlock(%struct.PlatformMutex* getelementptr inbounds (%struct.SpinLock* @_ZN3WTFL13pageheap_lockE, i32 0, i32 0)) nounwind
  %10 = getelementptr inbounds %"struct.WTF::TCMalloc_ThreadCache"* %heap.1, i32 0, i32 2
  %11 = load i8* %10, align 4
  %toBool15not = icmp eq i8 %11, 0                ; <i1> [#uses=1]
  br i1 %toBool15not, label %bb19, label %bb22

bb19:                                             ; preds = %bb14
  %.b = load i1* @_ZN3WTFL10tsd_initedE.b, align 4 ; <i1> [#uses=1]
  br i1 %.b, label %bb21, label %bb22

bb21:                                             ; preds = %bb19
  store i8 1, i8* %10, align 4
  %12 = load i32* @_ZN3WTFL8heap_keyE, align 4
  %13 = bitcast %"struct.WTF::TCMalloc_ThreadCache"* %heap.1 to i8*
  %14 = tail call  i32 @pthread_setspecific(i32 %12, i8* %13) nounwind
  ret %"struct.WTF::TCMalloc_ThreadCache"* %heap.1

bb22:                                             ; preds = %bb19, %bb14
  ret %"struct.WTF::TCMalloc_ThreadCache"* %heap.1
}

declare i32 @pthread_mutex_lock(%struct.PlatformMutex*)

declare i32 @pthread_mutex_unlock(%struct.PlatformMutex*)

declare hidden %"struct.WTF::TCMalloc_ThreadCache"* @_ZN3WTF20TCMalloc_ThreadCache7NewHeapEP17_opaque_pthread_t(%struct._opaque_pthread_t*) nounwind

declare i32 @pthread_setspecific(i32, i8*)

declare %struct._opaque_pthread_t* @pthread_self()

declare i32 @pthread_equal(%struct._opaque_pthread_t*, %struct._opaque_pthread_t*)

