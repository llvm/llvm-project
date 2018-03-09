; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
; Test to check that unused argument 'this' is not undefined in debug info.

target triple = "x86_64-apple-darwin10.2"

%struct.foo = type { i32 }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.foo*, i32)* @_ZN3foo3bazEi to i8*)], section "llvm.metadata"

; Function Attrs: noinline nounwind optsize readnone ssp
define i32 @_ZN3foo3bazEi(%struct.foo* nocapture %this, i32 %x) #0 align 2 !dbg !4 {
entry: 
  ; CHECK: DEBUG_VALUE: baz:this <- %rdi{{$}}
  tail call void @llvm.dbg.value(metadata %struct.foo* %this, i64 0, metadata !13, metadata !16), !dbg !17
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !18, metadata !16), !dbg !17
  %0 = mul nsw i32 %x, 7, !dbg !19
  %1 = add nsw i32 %0, 1, !dbg !19
  ret i32 %1, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { noinline nounwind optsize readnone ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "4.2.1 LLVM build", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2)
!1 = !DIFile(filename: "foo.cp", directory: "/tmp/")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "baz", linkageName: "_ZN3foo3bazEi", scope: !5, file: !1, line: 15, type: !10, isLocal: false, isDefinition: true, scopeLine: 15, virtualIndex: 6, isOptimized: true, unit: !0)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", scope: !1, file: !1, line: 3, size: 32, align: 32, elements: !6)
!6 = !{!7, !9, !4}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !1, line: 8, baseType: !8, size: 32, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEi", scope: !5, file: !1, line: 11, type: !10, isLocal: false, isDefinition: true, scopeLine: 11, virtualIndex: 6, isOptimized: true, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{!8, !12, !8}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !1, file: !1, baseType: !5, size: 64, align: 64, flags: DIFlagArtificial)
!13 = !DILocalVariable(name: "this", arg: 1, scope: !4, file: !1, line: 15, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_const_type, scope: !1, file: !1, baseType: !15, size: 64, align: 64, flags: DIFlagArtificial)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !1, file: !1, baseType: !5, size: 64, align: 64)
!16 = !DIExpression()
!17 = !DILocation(line: 0, scope: !4)
!18 = !DILocalVariable(name: "x", arg: 2, scope: !4, file: !1, line: 15, type: !8)
!19 = !DILocation(line: 16, scope: !20)
!20 = distinct !DILexicalBlock(scope: !4, file: !1, line: 15)
