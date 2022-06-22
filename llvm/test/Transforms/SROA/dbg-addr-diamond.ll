; RUN: opt -use-dbg-addr -passes=sroa -S < %s | FileCheck %s

; ModuleID = '<stdin>'
source_filename = "newvars.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

%struct.Pair = type { i32, i32 }

@pair = internal global %struct.Pair zeroinitializer

; Function Attrs: nounwind uwtable
define void @if_else(i32 %cond, i32 %a, i32 %b) !dbg !8 {
entry:
  %p = alloca %struct.Pair, align 4
  %0 = bitcast ptr %p to ptr, !dbg !25
  call void @llvm.dbg.addr(metadata ptr %p, metadata !20, metadata !DIExpression()), !dbg !26
  store i32 %a, ptr %p, align 4, !dbg !28
  %y = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 1, !dbg !34
  store i32 %b, ptr %y, align 4, !dbg !35
  %tobool = icmp ne i32 %cond, 0, !dbg !37
  br i1 %tobool, label %if.then, label %if.else, !dbg !39

if.then:                                          ; preds = %entry
  store i32 0, ptr %p, align 4, !dbg !42
  %y2 = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 1, !dbg !43
  store i32 %a, ptr %y2, align 4, !dbg !44
  br label %if.end, !dbg !45

if.else:                                          ; preds = %entry
  store i32 %b, ptr %p, align 4, !dbg !48
  %y4 = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 1, !dbg !49
  store i32 0, ptr %y4, align 4, !dbg !50
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %1 = bitcast ptr %p to ptr, !dbg !51
  %2 = bitcast ptr @pair to ptr, !dbg !51
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 %1, i64 8, i1 false), !dbg !51
  ret void
}

; CHECK-LABEL: define void @if_else(i32 %cond, i32 %a, i32 %b)
; CHECK: entry:
; CHECK:   call void @llvm.dbg.value(metadata i32 %a, metadata ![[PVAR:[0-9]+]], metadata ![[XFRAG:DIExpression\(DW_OP_LLVM_fragment, 0, 32\)]])
; CHECK:   call void @llvm.dbg.value(metadata i32 %b, metadata ![[PVAR]], metadata ![[YFRAG:DIExpression\(DW_OP_LLVM_fragment, 32, 32\)]])
; CHECK: if.then:
; CHECK:   call void @llvm.dbg.value(metadata i32 0, metadata ![[PVAR]], metadata ![[XFRAG]])
; CHECK:   call void @llvm.dbg.value(metadata i32 %a, metadata ![[PVAR]], metadata ![[YFRAG]])
; CHECK: if.else:
; CHECK:   call void @llvm.dbg.value(metadata i32 %b, metadata ![[PVAR]], metadata ![[XFRAG]])
; CHECK:   call void @llvm.dbg.value(metadata i32 0, metadata ![[PVAR]], metadata ![[YFRAG]])
; CHECK: if.end:
; CHECK:   %p.sroa.4.0 = phi i32 [ %a, %if.then ], [ 0, %if.else ]
; CHECK:   %p.sroa.0.0 = phi i32 [ 0, %if.then ], [ %b, %if.else ]
; CHECK:   call void @llvm.dbg.value(metadata i32 %p.sroa.0.0, metadata ![[PVAR]], metadata ![[XFRAG]])
; CHECK:   call void @llvm.dbg.value(metadata i32 %p.sroa.4.0, metadata ![[PVAR]], metadata ![[YFRAG]])

; CHECK: ![[PVAR]] = !DILocalVariable(name: "p", {{.*}})

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.addr(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "newvars.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "if_else", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !16)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !14, !14, !14}
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair", file: !1, line: 1, size: 64, elements: !12)
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 1, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !11, file: !1, line: 1, baseType: !14, size: 32, offset: 32)
!16 = !{!17, !18, !19, !20}
!17 = !DILocalVariable(name: "b", arg: 3, scope: !8, file: !1, line: 2, type: !14)
!18 = !DILocalVariable(name: "a", arg: 2, scope: !8, file: !1, line: 2, type: !14)
!19 = !DILocalVariable(name: "cond", arg: 1, scope: !8, file: !1, line: 2, type: !14)
!20 = !DILocalVariable(name: "p", scope: !8, file: !1, line: 3, type: !11)
!22 = !DILocation(line: 2, column: 42, scope: !8)
!23 = !DILocation(line: 2, column: 35, scope: !8)
!24 = !DILocation(line: 2, column: 25, scope: !8)
!25 = !DILocation(line: 3, column: 3, scope: !8)
!26 = !DILocation(line: 3, column: 15, scope: !8)
!27 = !DILocation(line: 4, column: 5, scope: !8)
!28 = !DILocation(line: 4, column: 7, scope: !8)
!29 = !{!30, !31, i64 0}
!30 = !{!"Pair", !31, i64 0, !31, i64 4}
!31 = !{!"int", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C/C++ TBAA"}
!34 = !DILocation(line: 5, column: 5, scope: !8)
!35 = !DILocation(line: 5, column: 7, scope: !8)
!36 = !{!30, !31, i64 4}
!37 = !DILocation(line: 6, column: 7, scope: !38)
!38 = distinct !DILexicalBlock(scope: !8, file: !1, line: 6, column: 7)
!39 = !DILocation(line: 6, column: 7, scope: !8)
!40 = !DILocation(line: 7, column: 7, scope: !41)
!41 = distinct !DILexicalBlock(scope: !38, file: !1, line: 6, column: 13)
!42 = !DILocation(line: 7, column: 9, scope: !41)
!43 = !DILocation(line: 8, column: 7, scope: !41)
!44 = !DILocation(line: 8, column: 9, scope: !41)
!45 = !DILocation(line: 9, column: 3, scope: !41)
!46 = !DILocation(line: 10, column: 7, scope: !47)
!47 = distinct !DILexicalBlock(scope: !38, file: !1, line: 9, column: 10)
!48 = !DILocation(line: 10, column: 9, scope: !47)
!49 = !DILocation(line: 11, column: 7, scope: !47)
!50 = !DILocation(line: 11, column: 9, scope: !47)
!51 = !DILocation(line: 13, column: 10, scope: !8)
!52 = !{i64 0, i64 4, !53, i64 4, i64 4, !53}
!53 = !{!31, !31, i64 0}
!54 = !DILocation(line: 14, column: 1, scope: !8)
