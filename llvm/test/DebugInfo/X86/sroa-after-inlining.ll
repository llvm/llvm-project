; RUN: opt %s -sroa -instcombine -inline -instcombine -sroa -verify -S -o - | FileCheck %s
;
; This test checks that SROA pass processes debug info correctly if applied twice.
; Specifically, after SROA works first time, instcombine converts dbg.declare
; intrinsics into dbg.value. Inlining creates new opportunities for SROA,
; so it is called again. This time it does not handle correctly previously
; inserted dbg.value intrinsics: current SROA implementation while doing
; "Migrate debug information from the old alloca to the new alloca(s)" handles
; only dbg.declare intrinsic. In this case, original dbg.declare was lowered by
; instcombine pass into dbg.value. When it comes into SROA second time, all dbg.value
; intrinsics, inserted by instcombine pass before second SROA, just not updated
; (though SROA was done). The fix is to not lower dbg.declare for structures.

;
; Hand-reduced from this example (-g -O -mllvm -disable-llvm-optzns -gno-column-info):
;
; struct S1 {
;     int p1;
;
;     bool IsNull (  ) { return p1 == 0; }
; };
;
; S1 foo ( void );
;
; int bar (  ) {
;     S1 result = foo();
;
;     if (result.IsNull())
;         return 0;
;
;     return result.p1 + 1;
; }

; CHECK: _Z3barv
; CHECK: %[[RESULT:.*]] = call i32 @_Z3foov
; CHECK: llvm.dbg.value(metadata i32 %[[RESULT]], metadata [[METADATA_IDX1:![0-9]+]]
; CHECK: ret
; CHECK: DICompileUnit
; CHECK: [[METADATA_IDX1]] = !DILocalVariable(name: "result"

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S1 = type { i32 }

$_ZN2S16IsNullEv = comdat any

define dso_local i32 @_Z3barv() !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %result = alloca %struct.S1, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %0 = bitcast %struct.S1* %result to i8*, !dbg !21
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #5, !dbg !21
  call void @llvm.dbg.declare(metadata %struct.S1* %result, metadata !12, metadata !DIExpression()), !dbg !21
  %call = call i32 @_Z3foov(), !dbg !21
  %coerce.dive = getelementptr inbounds %struct.S1, %struct.S1* %result, i32 0, i32 0, !dbg !21
  store i32 %call, i32* %coerce.dive, align 4, !dbg !21
  %call1 = call zeroext i1 @_ZN2S16IsNullEv(%struct.S1* %result), !dbg !22
  br i1 %call1, label %if.then, label %if.end, !dbg !24

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4, !dbg !25
  store i32 1, i32* %cleanup.dest.slot, align 4
  br label %cleanup, !dbg !25

if.end:                                           ; preds = %entry
  %p1 = getelementptr inbounds %struct.S1, %struct.S1* %result, i32 0, i32 0, !dbg !26
  %1 = load i32, i32* %p1, align 4, !dbg !26
  %add = add nsw i32 %1, 1, !dbg !26
  store i32 %add, i32* %retval, align 4, !dbg !26
  store i32 1, i32* %cleanup.dest.slot, align 4
  br label %cleanup, !dbg !26

cleanup:                                          ; preds = %if.end, %if.then
  %2 = bitcast %struct.S1* %result to i8*, !dbg !32
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #5, !dbg !32
  %3 = load i32, i32* %retval, align 4, !dbg !32
  ret i32 %3, !dbg !32
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare dso_local i32 @_Z3foov()

define linkonce_odr dso_local zeroext i1 @_ZN2S16IsNullEv(%struct.S1* %this) #4 comdat align 2 !dbg !33 {
entry:
  %this.addr = alloca %struct.S1*, align 8
  store %struct.S1* %this, %struct.S1** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.S1** %this.addr, metadata !35, metadata !DIExpression()), !dbg !39
  %this1 = load %struct.S1*, %struct.S1** %this.addr, align 8
  %p1 = getelementptr inbounds %struct.S1, %struct.S1* %this1, i32 0, i32 0, !dbg !40
  %0 = load i32, i32* %p1, align 4, !dbg !40
  %cmp = icmp eq i32 %0, 0, !dbg !40
  ret i1 %cmp, !dbg !40
}

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "sroa-after-inlining.cpp", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "result", scope: !7, file: !1, line: 10, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S1", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS2S1")
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "p1", scope: !13, file: !1, line: 2, baseType: !10, size: 32)
!16 = !DISubprogram(name: "IsNull", linkageName: "_ZN2S16IsNullEv", scope: !13, file: !1, line: 4, type: !17, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!17 = !DISubroutineType(types: !18)
!18 = !{!19, !20}
!19 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DILocation(line: 10, scope: !7)
!22 = !DILocation(line: 12, scope: !23)
!23 = distinct !DILexicalBlock(scope: !7, file: !1, line: 12)
!24 = !DILocation(line: 12, scope: !7)
!25 = !DILocation(line: 13, scope: !23)
!26 = !DILocation(line: 15, scope: !7)
!32 = !DILocation(line: 16, scope: !7)
!33 = distinct !DISubprogram(name: "IsNull", linkageName: "_ZN2S16IsNullEv", scope: !13, file: !1, line: 4, type: !17, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !16, retainedNodes: !34)
!34 = !{!35}
!35 = !DILocalVariable(name: "this", arg: 1, scope: !33, type: !36, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!39 = !DILocation(line: 0, scope: !33)
!40 = !DILocation(line: 4, scope: !33)
