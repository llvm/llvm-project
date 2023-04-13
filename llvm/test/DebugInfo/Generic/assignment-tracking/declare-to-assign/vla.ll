; RUN: opt -S %s -passes=declare-to-assign -o - | FileCheck %s

;; Check declare-to-assign ignores VLA-backed variables (for now).
;; From C++ source:
;; __attribute__ ((nodebug)) int sz;
;; void fun() {
;;   int x[sz];
;; }

; CHECK: llvm.dbg.declare(metadata ptr %vla, metadata ![[#]], metadata !DIExpression())

@sz = dso_local global i32 0, align 4

define dso_local void @_Z3funv() #0 !dbg !10 {
entry:
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  %0 = load i32, ptr @sz, align 4, !dbg !14
  %1 = zext i32 %0 to i64, !dbg !15
  %2 = call ptr @llvm.stacksave(), !dbg !15
  store ptr %2, ptr %saved_stack, align 8, !dbg !15
  %vla = alloca i32, i64 %1, align 16, !dbg !15
  store i64 %1, ptr %__vla_expr0, align 8, !dbg !15
  ;; Ignore this artificial variable for the purpose of the test.
  ; call void @llvm.dbg.declare(metadata ptr %__vla_expr0, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %vla, metadata !19, metadata !DIExpression()), !dbg !24
  %3 = load ptr, ptr %saved_stack, align 8, !dbg !25
  call void @llvm.stackrestore(ptr %3), !dbg !25
  ret void, !dbg !25
}

declare ptr @llvm.stacksave() #1
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2
declare void @llvm.stackrestore(ptr) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{}
!14 = !DILocation(line: 3, column: 9, scope: !10)
!15 = !DILocation(line: 3, column: 3, scope: !10)
!16 = !DILocalVariable(name: "__vla_expr0", scope: !10, type: !17, flags: DIFlagArtificial)
!17 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!18 = !DILocation(line: 0, scope: !10)
!19 = !DILocalVariable(name: "x", scope: !10, file: !1, line: 3, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, elements: !22)
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !{!23}
!23 = !DISubrange(count: !16)
!24 = !DILocation(line: 3, column: 7, scope: !10)
!25 = !DILocation(line: 4, column: 1, scope: !10)
