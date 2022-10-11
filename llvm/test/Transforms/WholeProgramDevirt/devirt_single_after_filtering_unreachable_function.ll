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
%Base = type { ptr }

@_ZTV7Derived = constant { [3 x ptr] } { [3 x ptr] [ ptr null, ptr null, ptr @_ZN7DerivedD0Ev] }, !type !0, !type !1, !type !2, !type !3
@_ZTV4Base = constant { [3 x ptr] } { [3 x ptr] [ ptr null, ptr null, ptr @_ZN4BaseD0Ev] }, !type !0, !type !1

declare i1 @llvm.type.test(ptr, metadata)

declare void @llvm.assume(i1)

define i32 @func(ptr %b) {
entry:
  %vtable = load ptr, ptr %b, !dbg !11
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base"), !dbg !11
  tail call void @llvm.assume(i1 %0), !dbg !11
  %1 = load ptr, ptr %vtable, !dbg !11
  tail call void %1(ptr %b), !dbg !11
  ret i32 0
}

define void @_ZN7DerivedD0Ev(ptr %this) {
entry:
  ret void
}

define void @_ZN4BaseD0Ev(ptr %this) {
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
