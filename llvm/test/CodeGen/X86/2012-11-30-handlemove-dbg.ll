; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test LiveInterval update handling of DBG_VALUE.
; rdar://12777252.
;
; CHECK: %entry
; CHECK: DEBUG_VALUE: subdivp:hg
; CHECK: j

%struct.node.0.27 = type { i16, double, [3 x double], i32, i32 }
%struct.hgstruct.2.29 = type { %struct.bnode.1.28*, [3 x double], double, [3 x double] }
%struct.bnode.1.28 = type { i16, double, [3 x double], i32, i32, [3 x double], [3 x double], [3 x double], double, %struct.bnode.1.28*, %struct.bnode.1.28* }

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define signext i16 @subdivp(%struct.node.0.27* nocapture %p, double %dsq, double %tolsq, %struct.hgstruct.2.29* nocapture byval align 8 %hg) nounwind uwtable readonly ssp !dbg !14 {
entry:
  call void @llvm.dbg.declare(metadata %struct.hgstruct.2.29* %hg, metadata !4, metadata !DIExpression()), !dbg !DILocation(scope: !14)
  %type = getelementptr inbounds %struct.node.0.27, %struct.node.0.27* %p, i64 0, i32 0
  %0 = load i16, i16* %type, align 2
  %cmp = icmp eq i16 %0, 1
  br i1 %cmp, label %return, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  %arrayidx6.1 = getelementptr inbounds %struct.hgstruct.2.29, %struct.hgstruct.2.29* %hg, i64 0, i32 1, i64 1
  %cmp22 = fcmp olt double 0.000000e+00, %dsq
  %conv24 = zext i1 %cmp22 to i16
  br label %return

return:                                           ; preds = %for.cond.preheader, %entry
  %retval.0 = phi i16 [ %conv24, %for.cond.preheader ], [ 0, %entry ]
  ret i16 %retval.0
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 168918) (llvm/trunk 168920)", isOptimized: true, emissionKind: FullDebug, file: !11, enums: !2, retainedTypes: !2, globals: !2)
!2 = !{}
!4 = !DILocalVariable(name: "hg", line: 725, arg: 4, scope: !14, file: !5, type: !6)
!5 = !DIFile(filename: "MultiSource/Benchmarks/Olden/bh/newbh.c", directory: "MultiSource/Benchmarks/Olden/bh")
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "hgstruct", line: 492, file: !11, baseType: !7)
!7 = !DICompositeType(tag: DW_TAG_structure_type, line: 487, size: 512, align: 64, file: !11)
!11 = !DIFile(filename: "MultiSource/Benchmarks/Olden/bh/newbh.c", directory: "MultiSource/Benchmarks/Olden/bh")
!12 = !{i32 1, !"Debug Info Version", i32 3}
!14 = distinct !DISubprogram(name: "subdivp", isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !11, scope: !5, type: !15)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
