; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source:
;   struct tt;
;   struct s1 { const struct tt *mp; };
;   int test1(struct s1 *arg)
;   {
;     return  0;
;   }
;
;   struct tt { int m1; int m2; };
;   struct s2 { const struct tt m3; };
;   int test2(struct s2 *arg)
;   {
;     return arg->m3.m1;
;   }
; Compilation flags:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.s1 = type { %struct.tt* }
%struct.tt = type { i32, i32 }
%struct.s2 = type { %struct.tt }

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @test1(%struct.s1* nocapture readnone %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %struct.s1* %arg, metadata !22, metadata !DIExpression()), !dbg !23
  ret i32 0, !dbg !24
}

; Function Attrs: norecurse nounwind readonly
define dso_local i32 @test2(%struct.s2* nocapture readonly %arg) local_unnamed_addr #1 !dbg !25 {
entry:
  call void @llvm.dbg.value(metadata %struct.s2* %arg, metadata !33, metadata !DIExpression()), !dbg !34
  %m1 = getelementptr inbounds %struct.s2, %struct.s2* %arg, i64 0, i32 0, i32 0, !dbg !35
  %0 = load i32, i32* %m1, align 4, !dbg !35, !tbaa !36
  ret i32 %0, !dbg !42
}

; CHECK:        .long   0                       # BTF_KIND_CONST(id = 4)
; CHECK-NEXT:   .long   167772160               # 0xa000000
; CHECK-NEXT:   .long   10

; CHECK:        .long   60                      # BTF_KIND_STRUCT(id = 9)
; CHECK-NEXT:   .long   67108865                # 0x4000001
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   63
; CHECK-NEXT:   .long   4
; CHECK-NEXT:   .long   0                       # 0x0

; CHECK:        .long   66                      # BTF_KIND_STRUCT(id = 10)
; CHECK-NEXT:   .long   67108866                # 0x4000002
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   69
; CHECK-NEXT:   .long   6
; CHECK-NEXT:   .long   0                       # 0x0
; CHECK-NEXT:   .long   72
; CHECK-NEXT:   .long   6
; CHECK-NEXT:   .long   32                      # 0x20

; CHECK:        .ascii  "s2"                    # string offset=60
; CHECK:        .ascii  "m3"                    # string offset=63
; CHECK:        .ascii  "tt"                    # string offset=66
; CHECK:        .ascii  "m1"                    # string offset=69
; CHECK:        .ascii  "m2"                    # string offset=72

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 7cfd267c518aba226b34b7fbfe8db70000b22053)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/btf")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 7cfd267c518aba226b34b7fbfe8db70000b22053)"}
!7 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 3, type: !8, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 2, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "mp", scope: !12, file: !1, line: 2, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tt", file: !1, line: 8, size: 64, elements: !18)
!18 = !{!19, !20}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "m1", scope: !17, file: !1, line: 8, baseType: !10, size: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "m2", scope: !17, file: !1, line: 8, baseType: !10, size: 32, offset: 32)
!21 = !{!22}
!22 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 3, type: !11)
!23 = !DILocation(line: 0, scope: !7)
!24 = !DILocation(line: 5, column: 3, scope: !7)
!25 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 10, type: !26, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32)
!26 = !DISubroutineType(types: !27)
!27 = !{!10, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s2", file: !1, line: 9, size: 64, elements: !30)
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "m3", scope: !29, file: !1, line: 9, baseType: !16, size: 64)
!32 = !{!33}
!33 = !DILocalVariable(name: "arg", arg: 1, scope: !25, file: !1, line: 10, type: !28)
!34 = !DILocation(line: 0, scope: !25)
!35 = !DILocation(line: 12, column: 18, scope: !25)
!36 = !{!37, !39, i64 0}
!37 = !{!"s2", !38, i64 0}
!38 = !{!"tt", !39, i64 0, !39, i64 4}
!39 = !{!"int", !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C/C++ TBAA"}
!42 = !DILocation(line: 12, column: 3, scope: !25)
