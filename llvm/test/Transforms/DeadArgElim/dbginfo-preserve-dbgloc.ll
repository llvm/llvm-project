; RUN: opt -deadargelim -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.Channel = type { i32, i32 }

; Function Attrs: nounwind uwtable
define void @f2(i32 %m, i32 %n) #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %m, metadata !12, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 %n, metadata !13, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata %struct.Channel* null, metadata !14, metadata !DIExpression()), !dbg !23
  %call = call %struct.Channel* (...) @foo(), !dbg !24
  call void @llvm.dbg.value(metadata %struct.Channel* %call, metadata !14, metadata !DIExpression()), !dbg !23
  %cmp = icmp sgt i32 %m, 3, !dbg !25
  br i1 %cmp, label %if.then, label %if.end, !dbg !27

if.then:                                          ; preds = %entry
  %call1 = call zeroext i1 @f1(i1 zeroext true, %struct.Channel* %call), !dbg !28
  br label %if.end, !dbg !28

if.end:                                           ; preds = %if.then, %entry
  %cmp2 = icmp sgt i32 %n, %m, !dbg !29
  br i1 %cmp2, label %if.then3, label %if.end5, !dbg !31

if.then3:                                         ; preds = %if.end
  %call4 = call zeroext i1 @f1(i1 zeroext false, %struct.Channel* %call), !dbg !32
  br label %if.end5, !dbg !32

if.end5:                                          ; preds = %if.then3, %if.end
  ret void, !dbg !33
}

declare %struct.Channel* @foo(...) local_unnamed_addr #1

; Function Attrs: noinline nounwind uwtable
define internal zeroext i1 @f1(i1 zeroext %is_y, %struct.Channel* %str) #4 !dbg !34 {
entry:
  %frombool = zext i1 %is_y to i8
; CHECK: call void @llvm.dbg.value(metadata i1 %is_y, metadata !39, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i1 %is_y, metadata !39, metadata !DIExpression()), !dbg !42
; CHECK: call void @llvm.dbg.value(metadata %struct.Channel* %str, metadata !40, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata %struct.Channel* %str, metadata !40, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata %struct.Channel* null, metadata !41, metadata !DIExpression()), !dbg !44
  %tobool = icmp ne %struct.Channel* %str, null, !dbg !45
  br i1 %tobool, label %if.end, label %if.then, !dbg !47

if.then:                                          ; preds = %entry
  call void (...) @baa(), !dbg !48
  br label %cleanup, !dbg !50

if.end:                                           ; preds = %entry
  %call = call %struct.Channel* (...) @foo(), !dbg !51
  call void @llvm.dbg.value(metadata %struct.Channel* %call, metadata !41, metadata !DIExpression()), !dbg !44
  %tobool1 = trunc i8 %frombool to i1, !dbg !52
  br i1 %tobool1, label %if.then2, label %if.end3, !dbg !56

if.then2:                                         ; preds = %if.end
  call void (...) @baa(), !dbg !57
  br label %cleanup, !dbg !56

if.end3:                                          ; preds = %if.end
  br label %cleanup, !dbg !56

cleanup:                                          ; preds = %if.end3, %if.then2, %if.then
  %retval.0 = phi i1 [ false, %if.then2 ], [ true, %if.end3 ], [ false, %if.then ]
  ret i1 %retval.0, !dbg !56
}

declare void @baa(...) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/dir")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0"}
!7 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 31, type: !8, isLocal: false, isDefinition: true, scopeLine: 32, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "m", arg: 1, scope: !7, file: !1, line: 31, type: !10)
!13 = !DILocalVariable(name: "n", arg: 2, scope: !7, file: !1, line: 31, type: !10)
!14 = !DILocalVariable(name: "str3", scope: !7, file: !1, line: 33, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "channel", file: !1, line: 6, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Channel", file: !1, line: 3, size: 64, elements: !18)
!18 = !{!19, !20}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !17, file: !1, line: 4, baseType: !10, size: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !17, file: !1, line: 5, baseType: !10, size: 32, offset: 32)
!21 = !DILocation(line: 31, column: 13, scope: !7)
!22 = !DILocation(line: 31, column: 20, scope: !7)
!23 = !DILocation(line: 33, column: 11, scope: !7)
!24 = !DILocation(line: 34, column: 9, scope: !7)
!25 = !DILocation(line: 36, column: 8, scope: !26)
!26 = distinct !DILexicalBlock(scope: !7, file: !1, line: 36, column: 6)
!27 = !DILocation(line: 36, column: 6, scope: !7)
!28 = !DILocation(line: 37, column: 3, scope: !26)
!29 = !DILocation(line: 39, column: 8, scope: !30)
!30 = distinct !DILexicalBlock(scope: !7, file: !1, line: 39, column: 6)
!31 = !DILocation(line: 39, column: 6, scope: !7)
!32 = !DILocation(line: 40, column: 3, scope: !30)
!33 = !DILocation(line: 41, column: 1, scope: !7)
!34 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 12, type: !35, isLocal: true, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !38)
!35 = !DISubroutineType(types: !36)
!36 = !{!37, !37, !15}
!37 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!38 = !{!39, !40, !41}
!39 = !DILocalVariable(name: "is_y", arg: 1, scope: !34, file: !1, line: 12, type: !37)
!40 = !DILocalVariable(name: "str", arg: 2, scope: !34, file: !1, line: 12, type: !15)
!41 = !DILocalVariable(name: "str2", scope: !34, file: !1, line: 14, type: !15)
!42 = !DILocation(line: 12, column: 21, scope: !34)
!43 = !DILocation(line: 12, column: 36, scope: !34)
!44 = !DILocation(line: 14, column: 11, scope: !34)
!45 = !DILocation(line: 16, column: 7, scope: !46)
!46 = distinct !DILexicalBlock(scope: !34, file: !1, line: 16, column: 6)
!47 = !DILocation(line: 16, column: 6, scope: !34)
!48 = !DILocation(line: 17, column: 3, scope: !49)
!49 = distinct !DILexicalBlock(scope: !46, file: !1, line: 16, column: 11)
!50 = !DILocation(line: 18, column: 3, scope: !49)
!51 = !DILocation(line: 21, column: 9, scope: !34)
!52 = !DILocation(line: 23, column: 6, scope: !34)
!53 = !DILocation(line: 24, column: 3, scope: !54)
!54 = distinct !DILexicalBlock(scope: !55, file: !1, line: 23, column: 11)
!55 = distinct !DILexicalBlock(scope: !34, file: !1, line: 23, column: 6)
!56 = !DILocation(line: 25, column: 3, scope: !54)
!57 = !DILocation(line: 28, column: 2, scope: !34)
