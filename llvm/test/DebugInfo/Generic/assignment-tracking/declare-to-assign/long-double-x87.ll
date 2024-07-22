; RUN: opt %s -S -passes=declare-to-assign -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators %s -S -passes=declare-to-assign -o - | FileCheck %s

;; Generated from this C++:
;; long double get();
;; void fun() {
;;   long double f;
;; }

;; Check a variable that is larger (128 bits according to !15) than its backing
;; alloca (80 bits) can be represented with assignment tracking. Create a
;; fragment for the dbg.assign for bits 0-80.

; CHECK: #dbg_assign(i1 undef, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 80), ![[#]], ptr %f, !DIExpression(),

define dso_local void @_Z3funv() #0 !dbg !10 {
entry:
  %f = alloca x86_fp80, align 16
  call void @llvm.dbg.declare(metadata ptr %f, metadata !14, metadata !DIExpression()), !dbg !16
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{}
!14 = !DILocalVariable(name: "f", scope: !10, file: !1, line: 3, type: !15)
!15 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!16 = !DILocation(line: 3, column: 15, scope: !10)
