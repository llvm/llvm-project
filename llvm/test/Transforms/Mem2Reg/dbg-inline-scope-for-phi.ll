; RUN: opt -S < %s -passes='function(mem2reg),require<verify>' | FileCheck %s
; RUN: opt -S < %s -passes='function(mem2reg),require<verify>' --try-experimental-debuginfo-iterators | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

; Original source (with some whitespace removed):
;
;   extern int *getp();
;   extern int cond();
;   int get1() { return *getp(); }
;   int get2(int *p) { return *p; }
;   int bug(int *p) {
;     if (cond()) return get1();
;     else return get2(p);
;   }

define i32 @get1() !dbg !8 {
  %1 = call ptr (...) @getp(), !dbg !12
  %2 = load i32, ptr %1, align 4, !dbg !13
  ret i32 %2, !dbg !14
}

declare ptr @getp(...)

define i32 @get2(ptr) !dbg !15 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !19, metadata !DIExpression()), !dbg !20
  %3 = load ptr, ptr %2, align 8, !dbg !21
  %4 = load i32, ptr %3, align 4, !dbg !22
  ret i32 %4, !dbg !23
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

; CHECK-LABEL: define i32 @bug
define i32 @bug(ptr) !dbg !24 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  call void @llvm.dbg.declare(metadata ptr %3, metadata !25, metadata !DIExpression()), !dbg !26
  %4 = call i32 (...) @cond(), !dbg !27
  %5 = icmp ne i32 %4, 0, !dbg !27
  br i1 %5, label %6, label %8, !dbg !29

; <label>:6:                                      ; preds = %1
  %7 = call i32 @get1(), !dbg !30
  store i32 %7, ptr %2, align 4, !dbg !31
  br label %11, !dbg !31

; <label>:8:                                      ; preds = %1
  %9 = load ptr, ptr %3, align 8, !dbg !32
  %10 = call i32 @get2(ptr %9), !dbg !33
  store i32 %10, ptr %2, align 4, !dbg !34
  br label %11, !dbg !34

; <label>:11:                                     ; preds = %8, %6
  %12 = load i32, ptr %2, align 4, !dbg !35
  ret i32 %12, !dbg !35

  ; CHECK: [[phi:%.*]] = phi i32 [ {{.*}} ], [ {{.*}} ], !dbg [[mergedLoc:![0-9]+]]
  ; CHECK-NEXT: ret i32 [[phi]], !dbg [[retLoc:![0-9]+]]
}

; CHECK: [[commonScope:![0-9]+]] = distinct !DILexicalBlock(scope: {{.*}}, file: !1, line: 15, column: 7)
; CHECK: [[mergedLoc]] = !DILocation(line: 0, scope: [[commonScope]])
; CHECK: [[retLoc]] = !DILocation(line: 23, column: 1

declare i32 @cond(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Apple LLVM version 9.1.0 (clang-902.2.37.2)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "bug.c", directory: "/bug")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"Apple LLVM version 9.1.0 (clang-902.2.37.2)"}
!8 = distinct !DISubprogram(name: "get1", scope: !1, file: !1, line: 6, type: !9, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 7, column: 11, scope: !8)
!13 = !DILocation(line: 7, column: 10, scope: !8)
!14 = !DILocation(line: 7, column: 3, scope: !8)
!15 = distinct !DISubprogram(name: "get2", scope: !1, file: !1, line: 10, type: !16, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!11, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!19 = !DILocalVariable(name: "p", arg: 1, scope: !15, file: !1, line: 10, type: !18)
!20 = !DILocation(line: 10, column: 15, scope: !15)
!21 = !DILocation(line: 11, column: 11, scope: !15)
!22 = !DILocation(line: 11, column: 10, scope: !15)
!23 = !DILocation(line: 11, column: 3, scope: !15)
!24 = distinct !DISubprogram(name: "bug", scope: !1, file: !1, line: 14, type: !16, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!25 = !DILocalVariable(name: "p", arg: 1, scope: !24, file: !1, line: 14, type: !18)
!26 = !DILocation(line: 14, column: 14, scope: !24)
!27 = !DILocation(line: 15, column: 7, scope: !28)
!28 = distinct !DILexicalBlock(scope: !24, file: !1, line: 15, column: 7)
!29 = !DILocation(line: 15, column: 7, scope: !24)
!30 = !DILocation(line: 16, column: 12, scope: !28)
!31 = !DILocation(line: 16, column: 5, scope: !28)
!32 = !DILocation(line: 18, column: 17, scope: !28)
!33 = !DILocation(line: 18, column: 12, scope: !28)
!34 = !DILocation(line: 18, column: 5, scope: !28)
!35 = !DILocation(line: 23, column: 1, scope: !24)
