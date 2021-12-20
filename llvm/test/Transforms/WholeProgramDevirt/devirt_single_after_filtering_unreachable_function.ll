; Test that regular LTO will analyze IR, detect unreachable functions and discard unreachable functions
; when finding virtual call targets.
; In this test case, the unreachable function is the virtual deleting destructor of an abstract class.

; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

; CHECK: remark: tmp.cc:21:3: single-impl: devirtualized a call to _ZN7DerivedD0Ev
; CHECK: remark: <unknown>:0:0: devirtualized _ZN7DerivedD0Ev

source_filename = "tmp.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Derived = type { %Base }
%Base = type { i32 (...)** }

@_ZTV7Derived = constant { [3 x i8*] } { [3 x i8*] [ i8* null, i8* null, i8* bitcast (void (%Derived*)* @_ZN7DerivedD0Ev to i8*)] }, !type !0, !type !1, !type !2, !type !3
@_ZTV4Base = constant { [3 x i8*] } { [3 x i8*] [ i8* null, i8* null, i8* bitcast (void (%Base*)* @_ZN4BaseD0Ev to i8*)] }, !type !0, !type !1

declare i1 @llvm.type.test(i8*, metadata)

declare void @llvm.assume(i1)

define i32 @func(%Base* %b) {
entry:
  %0 = bitcast %Base* %b to void (%Base*)***, !dbg !11
  %vtable = load void (%Base*)**, void (%Base*)*** %0, !dbg !11
  %1 = bitcast void (%Base*)** %vtable to i8*, !dbg !11
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS4Base"), !dbg !11
  tail call void @llvm.assume(i1 %2), !dbg !11
  %vfn = getelementptr inbounds void (%Base*)*, void (%Base*)** %vtable, i64 0, !dbg !11
  %3 = load void (%Base*)*, void (%Base*)** %vfn, !dbg !11
  tail call void %3(%Base* %b), !dbg !11
  ret i32 0
}

define void @_ZN7DerivedD0Ev(%Derived* %this) {
entry:
  ret void
}

define void @_ZN4BaseD0Ev(%Base* %this) {
entry:
  tail call void @llvm.trap()
  unreachable
}

declare void @llvm.trap()

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!7}

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 32, !"_ZTSM4BaseFvvE.virtual"}
!2 = !{i64 16, !"_ZTS7Derived"}
!3 = !{i64 32, !"_ZTSM7DerivedFvvE.virtual"}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !6)
!6 = !DIFile(filename: "tmp.cc", directory: "")
!7 = !{i32 2, !"Debug Info Version", i32 3}
!10= distinct !DISubprogram(name: "func", scope: !6, file: !6, unit: !5)
!11 = !DILocation(line: 21, column: 3, scope: !10)
