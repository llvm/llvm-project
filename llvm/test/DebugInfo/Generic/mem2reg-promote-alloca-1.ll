; RUN: opt -passes=mem2reg %s -S -o - | FileCheck %s

;; Check that mem2reg removes dbg.value(%param.addr, DIExpression(DW_OP_deref...))
;; when promoting the alloca %param.addr.
;;
;; $ clang inlining.c -O2 -g -emit-llvm -S -o tmp.ll -Xclang -disable-llvm-passes
;; $ opt tmp.ll -o - -instcombine -inline -S
;; $ cat inlining.c
;; int g;
;; __attribute__((__always_inline__))
;; static void use(int* p) {
;;   g = *p;
;; }
;;
;; __attribute__((__noinline__))
;; void fun(int param) {
;;   use(&param);
;; }

; CHECK: define dso_local void @fun(i32 %param)
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %param, metadata ![[PARAM:[0-9]+]], metadata !DIExpression())
; CHECK-NOT: call void @llvm.dbg.value({{.*}}, metadata ![[PARAM]]
; CHECK: ![[PARAM]] = !DILocalVariable(name: "param",

@g = dso_local global i32 0, align 4, !dbg !0

define dso_local void @fun(i32 %param) !dbg !12 {
entry:
  %param.addr = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 %param, metadata !16, metadata !DIExpression()), !dbg !17
  store i32 %param, i32* %param.addr, align 4
  call void @llvm.dbg.value(metadata i32* %param.addr, metadata !16, metadata !DIExpression(DW_OP_deref)), !dbg !17
  call void @llvm.dbg.value(metadata i32* %param.addr, metadata !22, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32* %param.addr, metadata !22, metadata !DIExpression()), !dbg !28
  %0 = load i32, i32* %param.addr, align 4, !dbg !30
  store i32 %0, i32* @g, align 4, !dbg !31
  ret void, !dbg !32
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !6, line: 8, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "inlining.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "inlining.c", directory: "/")
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 12.0.0"}
!12 = distinct !DISubprogram(name: "fun", scope: !6, file: !6, line: 15, type: !13, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !7}
!15 = !{!16}
!16 = !DILocalVariable(name: "param", arg: 1, scope: !12, file: !6, line: 15, type: !7)
!17 = !DILocation(line: 0, scope: !12)
!22 = !DILocalVariable(name: "p", arg: 1, scope: !23, file: !6, line: 10, type: !26)
!23 = distinct !DISubprogram(name: "use", scope: !6, file: !6, line: 10, type: !24, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !27)
!24 = !DISubroutineType(types: !25)
!25 = !{null, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!27 = !{!22}
!28 = !DILocation(line: 0, scope: !23, inlinedAt: !29)
!29 = distinct !DILocation(line: 16, column: 3, scope: !12)
!30 = !DILocation(line: 11, column: 7, scope: !23, inlinedAt: !29)
!31 = !DILocation(line: 11, column: 5, scope: !23, inlinedAt: !29)
!32 = !DILocation(line: 17, column: 1, scope: !12)
!34 = !DISubroutineType(types: !35)
!35 = !{!7}
