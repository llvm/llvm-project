; RUN: llc -mtriple=x86_64-pc-linux -split-dwarf-file=input.dwo -O3 \
; RUN:     -function-sections -data-sections \
; RUN:     -relocation-model=pic -filetype=asm \
; RUN:     -generate-type-units -o - %s | \
; RUN:     FileCheck %s --check-prefix=CHECK-ASM
; RUN: llc -mtriple=x86_64-pc-linux -split-dwarf-file=input.dwo -O3 \
; RUN:     -function-sections -data-sections \
; RUN:     -relocation-model=pic -filetype=obj \
; RUN:     -generate-type-units -o - %s | \
; RUN:     llvm-readelf --sections - | \
; RUN:     FileCheck %s --check-prefix=CHECK-ELF
; Created from `clang++ -fxray-instrument -gsplit-dwarf -fdebug-types-section
; -ffunction-sections -fdata-sections -emit-llvm -S input.cc`:
; input.cc:
;
; class a {
;   int b();
; };
; int a::b() {
;   for (;;)
;     ;
; }
;
; In this test we want to make sure that the xray_instr_map section for
; `a::b()` is actually associated with the function's symbol instead of the
; .debug_types.dwo section.
;
; CHECK-ASM: xray_fn_idx,"awo",@progbits,_ZN1a1bEv{{$}}
;
; CHECK-ELF-DAG: [[FSECT:[0-9]+]]] .text._ZN1a1bEv PROGBITS
; CHECK-ELF-DAG: [{{.*}}] .debug_types.dwo PROGBITS
; CHECK-ELF-DAG: [{{.*}}] xray_instr_map PROGBITS {{.*}} {{.*}} {{.*}} {{.*}} AL [[FSECT]]
target triple = "x86_64-pc-linux"

%class.a = type { i8 }

; Function Attrs: nounwind readnone uwtable
define i32 @_ZN1a1bEv(%class.a* nocapture readnone) local_unnamed_addr #0 align 2 !dbg !8 {
  tail call void @llvm.dbg.value(metadata %class.a* %0, metadata !17, metadata !DIExpression()), !dbg !19
  br label %2, !dbg !20

; <label>:2:                                      ; preds = %2, %1
  br label %2, !dbg !21, !llvm.loop !25
}


; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone uwtable "xray-instruction-threshold"="200" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version trunk (trunk r312634)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true)
!1 = !DIFile(filename: "input.cc", directory: "/usr/local/google/home/dberris/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{!"clang version trunk (trunk r312634)"}
!8 = distinct !DISubprogram(name: "b", linkageName: "_ZN1a1bEv", scope: !9, file: !1, line: 4, type: !12, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !11, retainedNodes: !16)
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "a", file: !1, line: 1, size: 8, elements: !10, identifier: "_ZTS1a")
!10 = !{!11}
!11 = !DISubprogram(name: "b", linkageName: "_ZN1a1bEv", scope: !9, file: !1, line: 2, type: !12, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !{!17}
!17 = !DILocalVariable(name: "this", arg: 1, scope: !8, type: !18, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!19 = !DILocation(line: 0, scope: !8)
!20 = !DILocation(line: 5, column: 3, scope: !8)
!21 = !DILocation(line: 5, column: 3, scope: !22)
!22 = !DILexicalBlockFile(scope: !23, file: !1, discriminator: 2)
!23 = distinct !DILexicalBlock(scope: !24, file: !1, line: 5, column: 3)
!24 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5, column: 3)
!25 = distinct !{!25, !26, !27}
!26 = !DILocation(line: 5, column: 3, scope: !24)
!27 = !DILocation(line: 6, column: 5, scope: !24)
