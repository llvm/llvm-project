; RUN: llc < %s -mtriple=i386-PC-Win32 | FileCheck %s

%class.C = type { i8 }
%struct.S = type { i32 }
%struct.M = type { i32, i32 }

declare void @_ZN1CC1Ev(%class.C* %this) unnamed_addr nounwind align 2
declare x86_thiscallcc void @_ZNK1C5SmallEv(%struct.S* noalias sret %agg.result, %class.C* %this) nounwind align 2
declare x86_thiscallcc void @_ZNK1C6MediumEv(%struct.M* noalias sret %agg.result, %class.C* %this) nounwind align 2

define void @testv() nounwind {
; CHECK: testv:
; CHECK: leal 16(%esp), %esi
; CHECK-NEXT: movl	%esi, (%esp)
; CHECK-NEXT: calll _ZN1CC1Ev
; CHECK: leal 8(%esp), %eax
; CHECK-NEXT: movl %esi, %ecx
; CHECK-NEXT: calll _ZNK1C5SmallEv
entry:
  %c = alloca %class.C, align 1
  %tmp = alloca %struct.S, align 4
  call void @_ZN1CC1Ev(%class.C* %c)
  ; This call should put the return structure as a pointer
  ; into EAX instead of returning directly in EAX.  The this
  ; pointer should go into ECX
  call x86_thiscallcc void @_ZNK1C5SmallEv(%struct.S* sret %tmp, %class.C* %c)
  ret void
}

define void @test2v() nounwind {
; CHECK: test2v:
; CHECK: leal 16(%esp), %esi
; CHECK-NEXT: movl	%esi, (%esp)
; CHECK-NEXT: calll _ZN1CC1Ev
; CHECK: leal 8(%esp), %eax
; CHECK-NEXT: movl %esi, %ecx
; CHECK-NEXT: calll _ZNK1C6MediumEv
entry:
  %c = alloca %class.C, align 1
  %tmp = alloca %struct.M, align 4
  call void @_ZN1CC1Ev(%class.C* %c)
  ; This call should put the return structure as a pointer
  ; into EAX instead of returning directly in EAX/EDX.  The this
  ; pointer should go into ECX
  call x86_thiscallcc void @_ZNK1C6MediumEv(%struct.M* sret %tmp, %class.C* %c)
  ret void
}
