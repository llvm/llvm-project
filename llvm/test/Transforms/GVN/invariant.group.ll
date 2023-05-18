; RUN: opt < %s -passes=gvn -S | FileCheck %s

%struct.A = type { ptr }
@_ZTV1A = available_externally unnamed_addr constant [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv], align 8
@_ZTI1A = external constant ptr

@unknownPtr = external global i8

; CHECK-LABEL: define i8 @simple() {
define i8 @simple() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    call void @foo(ptr %ptr)

    %a = load i8, ptr %ptr, !invariant.group !0
    %b = load i8, ptr %ptr, !invariant.group !0
    %c = load i8, ptr %ptr, !invariant.group !0
; CHECK: ret i8 42
    ret i8 %a
}

; CHECK-LABEL: define i8 @optimizable1() {
define i8 @optimizable1() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
    %a = load i8, ptr %ptr, !invariant.group !0
    
    call void @foo(ptr %ptr2); call to use %ptr2
; CHECK: ret i8 42
    ret i8 %a
}

; CHECK-LABEL: define i8 @optimizable2() {
define i8 @optimizable2() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    call void @foo(ptr %ptr)
    
    store i8 13, ptr %ptr ; can't use this store with invariant.group
    %a = load i8, ptr %ptr 
    call void @bar(i8 %a) ; call to use %a
    
    call void @foo(ptr %ptr)
    %b = load i8, ptr %ptr, !invariant.group !0
    
; CHECK: ret i8 42
    ret i8 %b
}

; CHECK-LABEL: define i1 @proveEqualityForStrip(
define i1 @proveEqualityForStrip(ptr %a) {
; FIXME: The first call could be also removed by GVN. Right now
; DCE removes it. The second call is CSE'd with the first one.
; CHECK: %b1 = call ptr @llvm.strip.invariant.group.p0(ptr %a)
  %b1 = call ptr @llvm.strip.invariant.group.p0(ptr %a)
; CHECK-NOT: llvm.strip.invariant.group
  %b2 = call ptr @llvm.strip.invariant.group.p0(ptr %a)
  %r = icmp eq ptr %b1, %b2
; CHECK: ret i1 true
  ret i1 %r
}
; CHECK-LABEL: define i8 @unoptimizable1() {
define i8 @unoptimizable1() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr
    call void @foo(ptr %ptr)
    %a = load i8, ptr %ptr, !invariant.group !0
; CHECK: ret i8 %a
    ret i8 %a
}

; CHECK-LABEL: define void @indirectLoads() {
define void @indirectLoads() {
entry:
  %a = alloca ptr, align 8
  
  %call = call ptr @getPointer(ptr null) 
  call void @_ZN1AC1Ev(ptr %call)
  
; CHECK: %vtable = load {{.*}} !invariant.group
  %vtable = load ptr, ptr %call, align 8, !invariant.group !0
  %cmp.vtables = icmp eq ptr %vtable, getelementptr inbounds ([3 x ptr], ptr @_ZTV1A, i64 0, i64 2)
  call void @llvm.assume(i1 %cmp.vtables)
  
  store ptr %call, ptr %a, align 8
  %0 = load ptr, ptr %a, align 8

; CHECK: call void @_ZN1A3fooEv(
  %vtable1 = load ptr, ptr %0, align 8, !invariant.group !0
  %1 = load ptr, ptr %vtable1, align 8
  call void %1(ptr %0)
  %2 = load ptr, ptr %a, align 8

; CHECK: call void @_ZN1A3fooEv(
  %vtable2 = load ptr, ptr %2, align 8, !invariant.group !0
  %3 = load ptr, ptr %vtable2, align 8
  
  call void %3(ptr %2)
  %4 = load ptr, ptr %a, align 8
  
  %vtable4 = load ptr, ptr %4, align 8, !invariant.group !0
  %5 = load ptr, ptr %vtable4, align 8
; CHECK: call void @_ZN1A3fooEv(
  call void %5(ptr %4)
 
  %vtable5 = load ptr, ptr %call, align 8, !invariant.group !0
  %6 = load ptr, ptr %vtable5, align 8
; CHECK: call void @_ZN1A3fooEv(
  call void %6(ptr %4)
  
  ret void
}

; CHECK-LABEL: define void @combiningBitCastWithLoad() {
define void @combiningBitCastWithLoad() {
entry:
  %a = alloca ptr, align 8
  
  %call = call ptr @getPointer(ptr null) 
  call void @_ZN1AC1Ev(ptr %call)
  
; CHECK: %vtable = load {{.*}} !invariant.group
  %vtable = load ptr, ptr %call, align 8, !invariant.group !0
  %cmp.vtables = icmp eq ptr %vtable, getelementptr inbounds ([3 x ptr], ptr @_ZTV1A, i64 0, i64 2)
  
  store ptr %call, ptr %a, align 8
; CHECK-NOT: !invariant.group
  %0 = load ptr, ptr %a, align 8

  %vtable1 = load ptr, ptr %0, align 8, !invariant.group !0
  %1 = load ptr, ptr %vtable1, align 8
  call void %1(ptr %0)

  ret void
}

; CHECK-LABEL:define void @loadCombine() {
define void @loadCombine() {
enter:
  %ptr = alloca i8
  store i8 42, ptr %ptr
  call void @foo(ptr %ptr)
; CHECK: %[[A:.*]] = load i8, ptr %ptr, align 1, !invariant.group
  %a = load i8, ptr %ptr, !invariant.group !0
; CHECK-NOT: load
  %b = load i8, ptr %ptr, !invariant.group !0
; CHECK: call void @bar(i8 %[[A]])
  call void @bar(i8 %a)
; CHECK: call void @bar(i8 %[[A]])
  call void @bar(i8 %b)
  ret void
}

; CHECK-LABEL: define void @loadCombine1() {
define void @loadCombine1() {
enter:
  %ptr = alloca i8
  store i8 42, ptr %ptr
  call void @foo(ptr %ptr)
; CHECK: %[[D:.*]] = load i8, ptr %ptr, align 1, !invariant.group
  %c = load i8, ptr %ptr
; CHECK-NOT: load
  %d = load i8, ptr %ptr, !invariant.group !0
; CHECK: call void @bar(i8 %[[D]])
  call void @bar(i8 %c)
; CHECK: call void @bar(i8 %[[D]])
  call void @bar(i8 %d)
  ret void
}

; CHECK-LABEL: define void @loadCombine2() {    
define void @loadCombine2() {
enter:
  %ptr = alloca i8
  store i8 42, ptr %ptr
  call void @foo(ptr %ptr)
; CHECK: %[[E:.*]] = load i8, ptr %ptr, align 1, !invariant.group
  %e = load i8, ptr %ptr, !invariant.group !0
; CHECK-NOT: load
  %f = load i8, ptr %ptr
; CHECK: call void @bar(i8 %[[E]])
  call void @bar(i8 %e)
; CHECK: call void @bar(i8 %[[E]])
  call void @bar(i8 %f)
  ret void
}

; CHECK-LABEL: define void @loadCombine3() {
define void @loadCombine3() {
enter:
  %ptr = alloca i8
  store i8 42, ptr %ptr
  call void @foo(ptr %ptr)
; CHECK: %[[E:.*]] = load i8, ptr %ptr, align 1, !invariant.group
  %e = load i8, ptr %ptr, !invariant.group !0
; CHECK-NOT: load
  %f = load i8, ptr %ptr, !invariant.group !0
; CHECK: call void @bar(i8 %[[E]])
  call void @bar(i8 %e)
; CHECK: call void @bar(i8 %[[E]])
  call void @bar(i8 %f)
  ret void
}

; CHECK-LABEL: define i8 @unoptimizable2() {
define i8 @unoptimizable2() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr
    call void @foo(ptr %ptr)
    %a = load i8, ptr %ptr
    call void @foo(ptr %ptr)
    %b = load i8, ptr %ptr, !invariant.group !0
    
; CHECK: ret i8 %a
    ret i8 %a
}

; CHECK-LABEL: define i8 @unoptimizable3() {
define i8 @unoptimizable3() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    %ptr2 = call ptr @getPointer(ptr %ptr)
    %a = load i8, ptr %ptr2, !invariant.group !0
    
; CHECK: ret i8 %a
    ret i8 %a
}

; CHECK-LABEL: define i8 @optimizable4() {
define i8 @optimizable4() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK-NOT: load
    %a = load i8, ptr %ptr2, !invariant.group !0
    
; CHECK: ret i8 42
    ret i8 %a
}

; CHECK-LABEL: define i8 @volatile1() {
define i8 @volatile1() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    call void @foo(ptr %ptr)
    %a = load i8, ptr %ptr, !invariant.group !0
    %b = load volatile i8, ptr %ptr
; CHECK: call void @bar(i8 %b)
    call void @bar(i8 %b)

    %c = load volatile i8, ptr %ptr, !invariant.group !0
; FIXME: we could change %c to 42, preserving volatile load
; CHECK: call void @bar(i8 %c)
    call void @bar(i8 %c)
; CHECK: ret i8 42
    ret i8 %a
}

; CHECK-LABEL: define i8 @volatile2() {
define i8 @volatile2() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    call void @foo(ptr %ptr)
    %a = load i8, ptr %ptr, !invariant.group !0
    %b = load volatile i8, ptr %ptr
; CHECK: call void @bar(i8 %b)
    call void @bar(i8 %b)

    %c = load volatile i8, ptr %ptr, !invariant.group !0
; FIXME: we could change %c to 42, preserving volatile load
; CHECK: call void @bar(i8 %c)
    call void @bar(i8 %c)
; CHECK: ret i8 42
    ret i8 %a
}

; CHECK-LABEL: define i8 @fun() {
define i8 @fun() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
    call void @foo(ptr %ptr)

    %a = load i8, ptr %ptr, !invariant.group !0 ; Can assume that value under %ptr didn't change
; CHECK: call void @bar(i8 42)
    call void @bar(i8 %a)

    %newPtr = call ptr @getPointer(ptr %ptr) 
    %c = load i8, ptr %newPtr, !invariant.group !0 ; Can't assume anything, because we only have information about %ptr
; CHECK: call void @bar(i8 %c)
    call void @bar(i8 %c)
    
    %unknownValue = load i8, ptr @unknownPtr
; FIXME: Can assume that %unknownValue == 42
; CHECK: store i8 %unknownValue, ptr %ptr, align 1, !invariant.group !0
    store i8 %unknownValue, ptr %ptr, !invariant.group !0 

    %newPtr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK-NOT: load
    %d = load i8, ptr %newPtr2, !invariant.group !0
; CHECK: ret i8 %unknownValue
    ret i8 %d
}

; This test checks if invariant.group understands gep with zeros
; CHECK-LABEL: define void @testGEP0() {
define void @testGEP0() {
  %a = alloca %struct.A, align 8
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1A, i64 0, i64 2), ptr %a, align 8, !invariant.group !0
; CHECK: call void @_ZN1A3fooEv(ptr nonnull dereferenceable(8) %a)
  call void @_ZN1A3fooEv(ptr nonnull dereferenceable(8) %a) ; This call may change vptr
  %1 = load i8, ptr @unknownPtr, align 4
  %2 = icmp eq i8 %1, 0
  br i1 %2, label %_Z1gR1A.exit, label %3

; This should be devirtualized by invariant.group
  %4 = load ptr, ptr %a, align 8, !invariant.group !0
  %5 = load ptr, ptr %4, align 8
; CHECK: call void @_ZN1A3fooEv(ptr nonnull %a)
  call void %5(ptr nonnull %a)
  br label %_Z1gR1A.exit

_Z1gR1A.exit:                                     ; preds = %0, %3
  ret void
}

; Check if no optimizations are performed with global pointers.
; FIXME: we could do the optimizations if we would check if dependency comes
; from the same function.
; CHECK-LABEL: define void @testGlobal() {
define void @testGlobal() {
; CHECK:  %a = load i8, ptr @unknownPtr, align 1, !invariant.group !0
   %a = load i8, ptr @unknownPtr, !invariant.group !0
   call void @foo2(ptr @unknownPtr, i8 %a)
; CHECK:  %1 = load i8, ptr @unknownPtr, align 1, !invariant.group !0
   %1 = load i8, ptr @unknownPtr, !invariant.group !0
   call void @bar(i8 %1)

   call void @fooBit(ptr @unknownPtr, i1 1)
; Adding regex because of canonicalization of bitcasts
; CHECK: %2 = load i1, ptr {{.*}}, !invariant.group !0
   %2 = load i1, ptr @unknownPtr, !invariant.group !0
   call void @fooBit(ptr @unknownPtr, i1 %2)
; CHECK:  %3 = load i1, ptr {{.*}}, !invariant.group !0
   %3 = load i1, ptr @unknownPtr, !invariant.group !0
   call void @fooBit(ptr @unknownPtr, i1 %3)
   ret void
}
; And in the case it is not global
; CHECK-LABEL: define void @testNotGlobal() {
define void @testNotGlobal() {
   %a = alloca i8
   call void @foo(ptr %a)
; CHECK:  %b = load i8, ptr %a, align 1, !invariant.group !0
   %b = load i8, ptr %a, !invariant.group !0
   call void @foo2(ptr %a, i8 %b)

   %1 = load i8, ptr %a, !invariant.group !0
; CHECK: call void @bar(i8 %b)
   call void @bar(i8 %1)

   call void @fooBit(ptr %a, i1 1)
; CHECK: %1 = trunc i8 %b to i1
   %2 = load i1, ptr %a, !invariant.group !0
; CHECK-NEXT: call void @fooBit(ptr %a, i1 %1)
   call void @fooBit(ptr %a, i1 %2)
   %3 = load i1, ptr %a, !invariant.group !0
; CHECK-NEXT: call void @fooBit(ptr %a, i1 %1)
   call void @fooBit(ptr %a, i1 %3)
   ret void
}

; CHECK-LABEL: define void @handling_loops()
define void @handling_loops() {
  %a = alloca %struct.A, align 8
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1A, i64 0, i64 2), ptr %a, align 8, !invariant.group !0
  %1 = load i8, ptr @unknownPtr, align 4
  %2 = icmp sgt i8 %1, 0
  br i1 %2, label %.lr.ph.i, label %_Z2g2R1A.exit

.lr.ph.i:                                         ; preds = %0
  %3 = load i8, ptr @unknownPtr, align 4
  %4 = icmp sgt i8 %3, 1
  br i1 %4, label %._crit_edge.preheader, label %_Z2g2R1A.exit

._crit_edge.preheader:                            ; preds = %.lr.ph.i
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.preheader, %._crit_edge
  %5 = phi i8 [ %7, %._crit_edge ], [ 1, %._crit_edge.preheader ]
  %.pre = load ptr, ptr %a, align 8, !invariant.group !0
  %6 = load ptr, ptr %.pre, align 8
  ; CHECK: call void @_ZN1A3fooEv(ptr nonnull %a)
  call void %6(ptr nonnull %a) #3
  ; CHECK-NOT: call void %
  %7 = add nuw nsw i8 %5, 1
  %8 = load i8, ptr @unknownPtr, align 4
  %9 = icmp slt i8 %7, %8
  br i1 %9, label %._crit_edge, label %_Z2g2R1A.exit.loopexit

_Z2g2R1A.exit.loopexit:                           ; preds = %._crit_edge
  br label %_Z2g2R1A.exit

_Z2g2R1A.exit:                                    ; preds = %_Z2g2R1A.exit.loopexit, %.lr.ph.i, %0
  ret void
}


declare void @foo(ptr)
declare void @foo2(ptr, i8)
declare void @bar(i8)
declare ptr @getPointer(ptr)
declare void @_ZN1A3fooEv(ptr)
declare void @_ZN1AC1Ev(ptr)
declare void @fooBit(ptr, i1)

declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)


declare void @llvm.assume(i1 %cmp.vtables)


!0 = !{}
