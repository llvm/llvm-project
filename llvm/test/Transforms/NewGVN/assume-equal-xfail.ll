; XFAIL: *
; RUN: opt < %s -passes=newgvn -S | FileCheck %s

%struct.A = type { ptr }
@_ZTV1A = available_externally unnamed_addr constant [4 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv, ptr @_ZN1A3barEv], align 8
@_ZTI1A = external constant ptr

; Checks if indirect calls can be replaced with direct
; assuming that %vtable == @_ZTV1A (with alignment).
; Checking const propagation across other BBs
; CHECK-LABEL: define void @_Z1gb(

define void @_Z1gb(i1 zeroext %p) {
entry:
  %call = tail call noalias ptr @_Znwm(i64 8) #4
  tail call void @_ZN1AC1Ev(ptr %call) #1
  %vtable = load ptr, ptr %call, align 8
  %cmp.vtables = icmp eq ptr %vtable, getelementptr inbounds ([4 x ptr], ptr @_ZTV1A, i64 0, i64 2)
  tail call void @llvm.assume(i1 %cmp.vtables)
  br i1 %p, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = load ptr, ptr %vtable, align 8
  
  ; CHECK: call i32 @_ZN1A3fooEv(
  %call2 = tail call i32 %0(ptr %call) #1
  
  br label %if.end

if.else:                                          ; preds = %entry
  %vfn47 = getelementptr inbounds ptr, ptr %vtable, i64 1
  
  ; CHECK: call i32 @_ZN1A3barEv(
  %1 = load ptr, ptr %vfn47, align 8
  
  %call5 = tail call i32 %1(ptr %call) #1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Check integration with invariant.group handling
; CHECK-LABEL: define void @invariantGroupHandling(i1 zeroext %p) {
define void @invariantGroupHandling(i1 zeroext %p) {
entry:
  %call = tail call noalias ptr @_Znwm(i64 8) #4
  tail call void @_ZN1AC1Ev(ptr %call) #1
  %vtable = load ptr, ptr %call, align 8, !invariant.group !0
  %cmp.vtables = icmp eq ptr %vtable, getelementptr inbounds ([4 x ptr], ptr @_ZTV1A, i64 0, i64 2)
  tail call void @llvm.assume(i1 %cmp.vtables)
  br i1 %p, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = load ptr, ptr %vtable, align 8
  
; CHECK: call i32 @_ZN1A3fooEv(
  %call2 = tail call i32 %0(ptr %call) #1
  %vtable1 = load ptr, ptr %call, align 8, !invariant.group !0
  %call1 = load ptr, ptr %vtable1, align 8
; CHECK: call i32 @_ZN1A3fooEv(
  %callx = tail call i32 %call1(ptr %call) #1
  
  %vtable2 = load ptr, ptr %call, align 8, !invariant.group !0
  %call4 = load ptr, ptr %vtable2, align 8
; CHECK: call i32 @_ZN1A3fooEv(
  %cally = tail call i32 %call4(ptr %call) #1
  
  %vtable3 = load ptr, ptr %call, align 8, !invariant.group !0
  %vfun = load ptr, ptr %vtable3, align 8
; CHECK: call i32 @_ZN1A3fooEv(
  %unknown = tail call i32 %vfun(ptr %call) #1
  
  br label %if.end

if.else:                                          ; preds = %entry
  %vfn47 = getelementptr inbounds ptr, ptr %vtable, i64 1
  
  ; CHECK: call i32 @_ZN1A3barEv(
  %1 = load ptr, ptr %vfn47, align 8
  
  %call5 = tail call i32 %1(ptr %call) #1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}


; Checking const propagation in the same BB
; CHECK-LABEL: define i32 @main()

define i32 @main() {
entry:
  %call = tail call noalias ptr @_Znwm(i64 8) 
  tail call void @_ZN1AC1Ev(ptr %call) 
  %vtable = load ptr, ptr %call, align 8
  %cmp.vtables = icmp eq ptr %vtable, getelementptr inbounds ([4 x ptr], ptr @_ZTV1A, i64 0, i64 2)
  tail call void @llvm.assume(i1 %cmp.vtables)
  
  ; CHECK: call i32 @_ZN1A3fooEv(
  %0 = load ptr, ptr %vtable, align 8
  
  %call2 = tail call i32 %0(ptr %call)
  ret i32 0
}

; This tests checks const propatation with fcmp instruction.
; CHECK-LABEL: define float @_Z1gf(float %p)

define float @_Z1gf(float %p) {
entry:
  %p.addr = alloca float, align 4
  %f = alloca float, align 4
  store float %p, ptr %p.addr, align 4
  
  store float 3.000000e+00, ptr %f, align 4
  %0 = load float, ptr %p.addr, align 4
  %1 = load float, ptr %f, align 4
  %cmp = fcmp oeq float %1, %0 ; note const on lhs
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: ret float 3.000000e+00
  ret float %0
}

; CHECK-LABEL: define float @_Z1hf(float %p)

define float @_Z1hf(float %p) {
entry:
  %p.addr = alloca float, align 4
  store float %p, ptr %p.addr, align 4
  
  %0 = load float, ptr %p.addr, align 4
  %cmp = fcmp nnan ueq float %0, 3.000000e+00
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: ret float 3.000000e+00
  ret float %0
}

declare noalias ptr @_Znwm(i64)
declare void @_ZN1AC1Ev(ptr)
declare void @llvm.assume(i1)
declare i32 @_ZN1A3fooEv(ptr)
declare i32 @_ZN1A3barEv(ptr)

!0 = !{!"struct A"}
