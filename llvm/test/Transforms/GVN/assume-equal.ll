; RUN: opt < %s -gvn -S | FileCheck %s

%struct.A = type { i32 (...)** }
@_ZTV1A = available_externally unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast (i8** @_ZTI1A to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1A3fooEv to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1A3barEv to i8*)], align 8
@_ZTI1A = external constant i8*

; Checks if indirect calls can be replaced with direct
; assuming that %vtable == @_ZTV1A (with alignment).
; Checking const propagation across other BBs
; CHECK-LABEL: define void @_Z1gb(

define void @_Z1gb(i1 zeroext %p) {
entry:
  %call = tail call noalias i8* @_Znwm(i64 8) #4
  %0 = bitcast i8* %call to %struct.A*
  tail call void @_ZN1AC1Ev(%struct.A* %0) #1
  %1 = bitcast i8* %call to i8***
  %vtable = load i8**, i8*** %1, align 8
  %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV1A, i64 0, i64 2)
  tail call void @llvm.assume(i1 %cmp.vtables)
  br i1 %p, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %vtable1.cast = bitcast i8** %vtable to i32 (%struct.A*)**
  %2 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable1.cast, align 8
  
  ; CHECK: call i32 @_ZN1A3fooEv(
  %call2 = tail call i32 %2(%struct.A* %0) #1
  
  br label %if.end

if.else:                                          ; preds = %entry
  %vfn47 = getelementptr inbounds i8*, i8** %vtable, i64 1
  %vfn4 = bitcast i8** %vfn47 to i32 (%struct.A*)**
  
  ; CHECK: call i32 @_ZN1A3barEv(
  %3 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vfn4, align 8
  
  %call5 = tail call i32 %3(%struct.A* %0) #1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Check integration with invariant.group handling
; CHECK-LABEL: define void @invariantGroupHandling(i1 zeroext %p) {
define void @invariantGroupHandling(i1 zeroext %p) {
entry:
  %call = tail call noalias i8* @_Znwm(i64 8) #4
  %0 = bitcast i8* %call to %struct.A*
  tail call void @_ZN1AC1Ev(%struct.A* %0) #1
  %1 = bitcast i8* %call to i8***
  %vtable = load i8**, i8*** %1, align 8, !invariant.group !0
  %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV1A, i64 0, i64 2)
  tail call void @llvm.assume(i1 %cmp.vtables)
  br i1 %p, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %vtable1.cast = bitcast i8** %vtable to i32 (%struct.A*)**
  %2 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable1.cast, align 8
  
; CHECK: call i32 @_ZN1A3fooEv(
  %call2 = tail call i32 %2(%struct.A* %0) #1
  %vtable1 = load i8**, i8*** %1, align 8, !invariant.group !0
  %vtable2.cast = bitcast i8** %vtable1 to i32 (%struct.A*)**
  %call1 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable2.cast, align 8
; CHECK: call i32 @_ZN1A3fooEv(
  %callx = tail call i32 %call1(%struct.A* %0) #1
  
  %vtable2 = load i8**, i8*** %1, align 8, !invariant.group !0
  %vtable3.cast = bitcast i8** %vtable2 to i32 (%struct.A*)**
  %call4 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable3.cast, align 8
; CHECK: call i32 @_ZN1A3fooEv(
  %cally = tail call i32 %call4(%struct.A* %0) #1
  
  %b = bitcast i8* %call to %struct.A**
  %vtable3 = load %struct.A*, %struct.A** %b, align 8, !invariant.group !0
  %vtable4.cast = bitcast %struct.A* %vtable3 to i32 (%struct.A*)**
  %vfun = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable4.cast, align 8
; CHECK: call i32 @_ZN1A3fooEv(
  %unknown = tail call i32 %vfun(%struct.A* %0) #1
  
  br label %if.end

if.else:                                          ; preds = %entry
  %vfn47 = getelementptr inbounds i8*, i8** %vtable, i64 1
  %vfn4 = bitcast i8** %vfn47 to i32 (%struct.A*)**
  
  ; CHECK: call i32 @_ZN1A3barEv(
  %3 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vfn4, align 8
  
  %call5 = tail call i32 %3(%struct.A* %0) #1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}


; Checking const propagation in the same BB
; CHECK-LABEL: define i32 @main()

define i32 @main() {
entry:
  %call = tail call noalias i8* @_Znwm(i64 8) 
  %0 = bitcast i8* %call to %struct.A*
  tail call void @_ZN1AC1Ev(%struct.A* %0) 
  %1 = bitcast i8* %call to i8***
  %vtable = load i8**, i8*** %1, align 8
  %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV1A, i64 0, i64 2)
  tail call void @llvm.assume(i1 %cmp.vtables)
  %vtable1.cast = bitcast i8** %vtable to i32 (%struct.A*)**
  
  ; CHECK: call i32 @_ZN1A3fooEv(
  %2 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable1.cast, align 8
  
  %call2 = tail call i32 %2(%struct.A* %0)
  ret i32 0
}

; This tests checks const propatation with fcmp instruction.
; CHECK-LABEL: define float @_Z1gf(float %p)

define float @_Z1gf(float %p) {
entry:
  %p.addr = alloca float, align 4
  %f = alloca float, align 4
  store float %p, float* %p.addr, align 4
  
  store float 3.000000e+00, float* %f, align 4
  %0 = load float, float* %p.addr, align 4
  %1 = load float, float* %f, align 4
  %cmp = fcmp oeq float %1, %0 ; note const on lhs
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: ret float 3.000000e+00
  ret float %0
}

; CHECK-LABEL: define float @_Z1hf(float %p)

define float @_Z1hf(float %p) {
entry:
  %p.addr = alloca float, align 4
  store float %p, float* %p.addr, align 4
  
  %0 = load float, float* %p.addr, align 4
  %cmp = fcmp nnan ueq float %0, 3.000000e+00
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: ret float 3.000000e+00
  ret float %0
}

; CHECK-LABEL: define float @_Z1if(float %p)
define float @_Z1if(float %p) {
entry:
  %p.addr = alloca float, align 4
  store float %p, float* %p.addr, align 4
  
  %0 = load float, float* %p.addr, align 4
  %cmp = fcmp ueq float %0, 3.000000e+00 ; no nnan flag - can't propagate
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK-NOT: ret float 3.000000e+00
  ret float %0
}

; This test checks if constant propagation works for multiple node edges
; CHECK-LABEL: define i32 @_Z1ii(i32 %p)
define i32 @_Z1ii(i32 %p) {
entry:
  %cmp = icmp eq i32 %p, 42
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: br i1 true, label %bb2, label %bb2
  br i1 %cmp, label %bb2, label %bb2
bb2:
  call void @llvm.assume(i1 true)
  ; CHECK: br i1 true, label %bb2, label %bb2
  br i1 %cmp, label %bb2, label %bb2
  
  ; CHECK: ret i32 42
  ret i32 %p
}

; CHECK-LABEL: define i32 @_Z1ij(i32 %p)
define i32 @_Z1ij(i32 %p) {
entry:
  %cmp = icmp eq i32 %p, 42
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: br i1 true, label %bb2, label %bb2
  br i1 %cmp, label %bb2, label %bb2
bb2:
   ; CHECK-NOT: %cmp2 = 
  %cmp2 = icmp eq i32 %p, 42
  ; CHECK-NOT: call void @llvm.assume(
  call void @llvm.assume(i1 %cmp2)
  
  ; CHECK: br i1 true, label %bb2, label %bb2
  br i1 %cmp, label %bb2, label %bb2
  
  ; CHECK: ret i32 42
  ret i32 %p
}

; CHECK-LABEL: define i32 @_Z1ik(i32 %p)
define i32 @_Z1ik(i32 %p) {
entry:
  %cmp = icmp eq i32 %p, 42
  call void @llvm.assume(i1 %cmp)
  
  ; CHECK: br i1 true, label %bb2, label %bb3
  br i1 %cmp, label %bb2, label %bb3
bb2:
  ; CHECK-NOT: %cmp3 = 
  %cmp3 = icmp eq i32 %p, 43
  ; CHECK: store i8 undef, i8* null
  call void @llvm.assume(i1 %cmp3)
  ret i32 15
bb3:
  ret i32 17
}

; This test checks if GVN can do the constant propagation correctly
; when there are multiple uses of the same assume value in the 
; basic block that has a loop back-edge pointing to itself.
;
; CHECK-LABEL: define i32 @_Z1il(i32 %val, i1 %k)
define i32 @_Z1il(i32 %val, i1 %k) {
  br label %next

next:
; CHECK: tail call void @llvm.assume(i1 %k)
; CHECK-NEXT: %cmp = icmp eq i32 %val, 50
  tail call void @llvm.assume(i1 %k)
  tail call void @llvm.assume(i1 %k)
  %cmp = icmp eq i32 %val, 50
  br i1 %cmp, label %next, label %meh

meh:
  ret i32 0 
}

; This test checks if GVN can prevent the constant propagation correctly
; in the successor blocks that are not dominated by the basic block
; with the assume instruction.
;
; CHECK-LABEL: define i1 @_z1im(i32 %val, i1 %k, i1 %j)
define i1 @_z1im(i32 %val, i1 %k, i1 %j) {
  br i1 %j, label %next, label %meh

next:
; CHECK: tail call void @llvm.assume(i1 %k)
; CHECK-NEXT: br label %meh
  tail call void @llvm.assume(i1 %k)
  tail call void @llvm.assume(i1 %k)
  br label %meh

meh:
; CHECK: ret i1 %k
  ret i1 %k
}

declare noalias i8* @_Znwm(i64)
declare void @_ZN1AC1Ev(%struct.A*)
declare void @llvm.assume(i1)
declare i32 @_ZN1A3fooEv(%struct.A*)
declare i32 @_ZN1A3barEv(%struct.A*)

!0 = !{!"struct A"}
