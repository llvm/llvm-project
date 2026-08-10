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

%class.A = type { ptr }
%class.B = type { %class.A }

@_ZTV1A = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1A3fooEv] }, align 8, !type !0, !vcall_visibility !2
@_ZTV1B = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1B3fooEv] }, align 8, !type !0, !type !3, !vcall_visibility !2

declare void @_ZN1A3fooEv(ptr nocapture %this)

define hidden void @_ZN1B3fooEv(ptr nocapture %this) {
entry:
  ret void
}

; Function Attrs: nounwind readnone willreturn
declare i1 @llvm.type.test(ptr, metadata)

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1)

; Function Attrs: uwtable
define hidden void @_Z6callerP1B(ptr %b) local_unnamed_addr {
entry:
  %vtable = load ptr, ptr %b, align 8, !tbaa !12, !invariant.group !15
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1B")
  tail call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable, align 8, !invariant.load !15
  tail call void %1(ptr %b)
  %2 = tail call i1 @llvm.type.test(ptr nonnull %vtable, metadata !"_ZTS1A")
  tail call void @llvm.assume(i1 %2)
  tail call void %1(ptr %b)
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
