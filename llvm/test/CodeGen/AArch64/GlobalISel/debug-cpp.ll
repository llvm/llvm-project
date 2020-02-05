; RUN: llc -global-isel -mtriple=aarch64 %s -stop-after=irtranslator -o - | FileCheck %s
; RUN: llc -mtriple=aarch64 -global-isel --global-isel-abort=0 -o /dev/null

; struct NTCopy {
;   NTCopy();
;   NTCopy(const NTCopy &);
;   int x;
; };
; int foo(NTCopy o) {
;   return o.x;
; }

; ModuleID = 'ntcopy.cpp'
source_filename = "ntcopy.cpp"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.NTCopy = type { i32 }

; CHECK-LABEL: name: _Z3foo6NTCopy
; CHECK: DBG_VALUE %{{[0-9]+}}(p0), 0, !23, !DIExpression(), debug-location !24
; Function Attrs: noinline nounwind optnone
define dso_local i32 @_Z3foo6NTCopy(%struct.NTCopy* %o) #0 !dbg !7 {
entry:
  call void @llvm.dbg.declare(metadata %struct.NTCopy* %o, metadata !23, metadata !DIExpression()), !dbg !24
  %x = getelementptr inbounds %struct.NTCopy, %struct.NTCopy* %o, i32 0, i32 0, !dbg !25
  %0 = load i32, i32* %x, align 4, !dbg !25
  ret i32 %0, !dbg !26
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "ntcopy.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 "}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foo6NTCopy", scope: !1, file: !1, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NTCopy", file: !1, line: 1, size: 32, flags: DIFlagTypePassByReference, elements: !12, identifier: "_ZTS6NTCopy")
!12 = !{!13, !14, !18}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 4, baseType: !10, size: 32)
!14 = !DISubprogram(name: "NTCopy", scope: !11, file: !1, line: 2, type: !15, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DISubprogram(name: "NTCopy", scope: !11, file: !1, line: 3, type: !19, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !17, !21}
!21 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!23 = !DILocalVariable(name: "o", arg: 1, scope: !7, file: !1, line: 6, type: !11)
!24 = !DILocation(line: 6, column: 16, scope: !7)
!25 = !DILocation(line: 7, column: 12, scope: !7)
!26 = !DILocation(line: 7, column: 3, scope: !7)
