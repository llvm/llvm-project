; RUN: llc -filetype=asm < %s
;
; We check that we don't crash when printing assembly comments that include
; a DW_OP_bit_piece
;
; Regenerate from
; void fn1() {
; struct {
;   int dword[2];
; } u;
; u.dword[1] = 0;
; };
; via clang++ -g -fno-integrated-as -Os

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


%struct.anon = type { [2 x i32] }

; Function Attrs: norecurse nounwind optsize readnone uwtable
define void @_Z3fn1v() #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !8, metadata !19), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !8, metadata !21), !dbg !20
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { norecurse nounwind optsize readnone uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 256088) (llvm/trunk 256097)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/mnt/extra")
!2 = !{}
!4 = distinct !DISubprogram(name: "fn1", linkageName: "_Z3fn1v", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "u", scope: !4, file: !1, line: 4, type: !9)
!9 = !DICompositeType(tag: DW_TAG_structure_type, scope: !4, file: !1, line: 2, size: 64, align: 32, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "dword", scope: !9, file: !1, line: 3, baseType: !12, size: 64, align: 32)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 64, align: 32, elements: !14)
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 2)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.8.0 (trunk 256088) (llvm/trunk 256097)"}
!19 = !DIExpression()
!20 = !DILocation(line: 4, column: 5, scope: !4)
!21 = !DIExpression(DW_OP_LLVM_fragment, 32, 32)
!22 = !DILocation(line: 6, column: 1, scope: !4)
