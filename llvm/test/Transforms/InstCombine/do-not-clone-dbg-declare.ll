
; RUN: opt %s -instcombine -verify -S -o - | FileCheck %s

; Hand-reduced from this example.
; -g -O -mllvm -disable-llvm-optzns -gno-column-info
; plus opt -sroa -instcombine -inline

; #include <stdio.h>
;
; struct S1 {
;     int p1;
;     int p2;
;
;     bool IsNull (  ) {
;         return p1 == 0;
;     }
; };
;
; S1 foo ( void );
;
; int bar (  ) {
;
;     S1 result = foo();
;
;     if ( result.IsNull() )
;         return 0;
;
;     result.p1 = 2;
;     result.p2 = 3;
;
;     int* ptr = &result.p1;
;
;     printf("%d", *ptr);
;     printf("%d", *(ptr+1));
;
;     return result.p1 + 1;
; }

; CHECK: _Z3barv
; CHECK: llvm.dbg.declare(metadata i64* %{{.*}}, metadata [[METADATA_IDX1:![0-9]+]]
; CHECK-NOT: llvm.dbg.declare(metadata %struct.S1* %{{.*}}, metadata [[METADATA_IDX1]]
; CHECK: ret
; CHECK: DICompileUnit
; CHECK: [[METADATA_IDX1]] = !DILocalVariable(name: "result"

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S1 = type { i32, i32 }

@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1

define dso_local i32 @_Z3barv() !dbg !7 {
entry:
  %result = alloca i64, align 8
  %tmpcast = bitcast i64* %result to %struct.S1*
  %0 = bitcast i64* %result to i8*, !dbg !24
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #4, !dbg !24
  call void @llvm.dbg.declare(metadata %struct.S1* %tmpcast, metadata !12, metadata !DIExpression()), !dbg !24
  %call = call i64 @_Z3foov(), !dbg !24
  store i64 %call, i64* %result, align 8, !dbg !24
  call void @llvm.dbg.value(metadata %struct.S1* %tmpcast, metadata !25, metadata !DIExpression()), !dbg !29
  %p1.i = getelementptr inbounds %struct.S1, %struct.S1* %tmpcast, i64 0, i32 0, !dbg !32
  %1 = load i32, i32* %p1.i, align 4, !dbg !32
  %cmp.i = icmp eq i32 %1, 0, !dbg !32
  br i1 %cmp.i, label %if.then, label %if.end, !dbg !38

if.then:                                          ; preds = %entry
  br label %cleanup, !dbg !38

if.end:                                           ; preds = %entry

  %p1 = bitcast i64* %result to i32*, !dbg !38
  store i32 2, i32* %p1, align 8, !dbg !38
  %p2 = getelementptr inbounds %struct.S1, %struct.S1* %tmpcast, i64 0, i32 1, !dbg !38
  store i32 3, i32* %p2, align 4, !dbg !38
  %p12 = bitcast i64* %result to i32*, !dbg !38
  call void @llvm.dbg.value(metadata i32* %p12, metadata !22, metadata !DIExpression()), !dbg !38
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 2), !dbg !38
  %add.ptr = getelementptr inbounds i32, i32* %p12, i64 1, !dbg !38
  %2 = load i32, i32* %add.ptr, align 4, !dbg !38
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %2), !dbg !38
  %p15 = bitcast i64* %result to i32*, !dbg !38
  %3 = load i32, i32* %p15, align 8, !dbg !38
  %add = add nsw i32 %3, 1, !dbg !38
  br label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ %add, %if.end ], !dbg !38
  %4 = bitcast i64* %result to i8*, !dbg !38
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %4) #4, !dbg !38
  ret i32 %retval.0, !dbg !38
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare dso_local i64 @_Z3foov() #3

declare dso_local i32 @printf(i8*, ...) #3

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 15, type: !8, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !22}
!12 = !DILocalVariable(name: "result", scope: !7, file: !1, line: 17, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S1", file: !1, line: 4, size: 64, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS2S1")
!14 = !{!15, !16, !17}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "p1", scope: !13, file: !1, line: 5, baseType: !10, size: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "p2", scope: !13, file: !1, line: 6, baseType: !10, size: 32, offset: 32)
!17 = !DISubprogram(name: "IsNull", linkageName: "_ZN2S16IsNullEv", scope: !13, file: !1, line: 8, type: !18, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !21}
!20 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DILocalVariable(name: "ptr", scope: !7, file: !1, line: 25, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!24 = !DILocation(line: 17, scope: !7)
!25 = !DILocalVariable(name: "this", arg: 1, scope: !26, type: !28, flags: DIFlagArtificial | DIFlagObjectPointer)
!26 = distinct !DISubprogram(name: "IsNull", linkageName: "_ZN2S16IsNullEv", scope: !13, file: !1, line: 8, type: !18, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !17, retainedNodes: !27)
!27 = !{!25}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!29 = !DILocation(line: 0, scope: !26, inlinedAt: !30)
!30 = distinct !DILocation(line: 19, scope: !31)
!31 = distinct !DILexicalBlock(scope: !7, file: !1, line: 19)
!32 = !DILocation(line: 9, scope: !26, inlinedAt: !30)
!38 = !DILocation(line: 0, scope: !7)
