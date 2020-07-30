; Test to ensure that devirtualization will succeed when there is an earlier
; type test also corresponding to the same vtable (when indicated by invariant
; load metadata), that provides a more refined type. This could happen in
; after inlining into a caller passing a derived type.

; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto2 run -o %t.out %t.o \
; RUN:	 -pass-remarks=wholeprogramdevirt \
; RUN:	 -r %t.o,_ZN1A3fooEv,px \
; RUN:	 -r %t.o,_ZN1B3fooEv,px \
; RUN:	 -r %t.o,_Z6callerP1B,px \
; RUN:	 -r %t.o,_ZTV1A,px \
; RUN:	 -r %t.o,_ZTV1B,px \
; RUN:	 -save-temps 2>&1 | FileCheck %s

; CHECK-COUNT-2: single-impl: devirtualized a call to _ZN1B3fooEv

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR
; IR-NOT: tail call void %

; ModuleID = 'devirt_multiple_type_test.o'
source_filename = "devirt_multiple_type_test.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.A = type { i32 (...)** }
%class.B = type { %class.A }

@_ZTV1A = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* undef, i8* bitcast (void (%class.A*)* @_ZN1A3fooEv to i8*)] }, align 8, !type !0, !vcall_visibility !2
@_ZTV1B = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* undef, i8* bitcast (void (%class.B*)* @_ZN1B3fooEv to i8*)] }, align 8, !type !0, !type !3, !vcall_visibility !2

declare void @_ZN1A3fooEv(%class.A* nocapture %this)

define hidden void @_ZN1B3fooEv(%class.B* nocapture %this) {
entry:
  ret void
}

; Function Attrs: nounwind readnone willreturn
declare i1 @llvm.type.test(i8*, metadata)

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1)

; Function Attrs: uwtable
define hidden void @_Z6callerP1B(%class.B* %b) local_unnamed_addr {
entry:
  %0 = bitcast %class.B* %b to void (%class.B*)***
  %vtable = load void (%class.B*)**, void (%class.B*)*** %0, align 8, !tbaa !12, !invariant.group !15
  %1 = bitcast void (%class.B*)** %vtable to i8*
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1B")
  tail call void @llvm.assume(i1 %2)
  %3 = load void (%class.B*)*, void (%class.B*)** %vtable, align 8, !invariant.load !15
  tail call void %3(%class.B* %b)
  %4 = getelementptr %class.B, %class.B* %b, i64 0, i32 0
  %5 = bitcast void (%class.B*)** %vtable to i8*
  %6 = tail call i1 @llvm.type.test(i8* nonnull %5, metadata !"_ZTS1A")
  tail call void @llvm.assume(i1 %6)
  %7 = bitcast void (%class.B*)* %3 to void (%class.A*)*
  tail call void %7(%class.A* %4)
  ret void
}

!llvm.module.flags = !{!5, !6, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !{i64 16, !"_ZTS1A"}
!2 = !{i64 1}
!3 = !{i64 16, !"_ZTS1B"}
!5 = !{i32 1, !"StrictVTablePointers", i32 1}
!6 = !{i32 3, !"StrictVTablePointersRequirement", !7}
!7 = !{!"StrictVTablePointers", i32 1}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 1, !"Virtual Function Elim", i32 0}
!10 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!11 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 85247c1e898f88d65154b9a437b4bd83fcad8d52)"}
!12 = !{!13, !13, i64 0}
!13 = !{!"vtable pointer", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}
!15 = !{}
