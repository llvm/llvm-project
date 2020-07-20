; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source:
;   struct tt;
;   typedef struct tt _tt;
;   typedef _tt __tt;
;   struct s1 { __tt *mp; };
;   int test1(struct s1 *arg)
;   {
;     return  0;
;   }
;
;   struct tt { int m1; int m2; };
;   struct s2 { __tt m3; };
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
  call void @llvm.dbg.value(metadata %struct.s1* %arg, metadata !23, metadata !DIExpression()), !dbg !24
  ret i32 0, !dbg !25
}

; Function Attrs: norecurse nounwind readonly
define dso_local i32 @test2(%struct.s2* nocapture readonly %arg) local_unnamed_addr #1 !dbg !26 {
entry:
  call void @llvm.dbg.value(metadata %struct.s2* %arg, metadata !34, metadata !DIExpression()), !dbg !35
  %m1 = getelementptr inbounds %struct.s2, %struct.s2* %arg, i64 0, i32 0, i32 0, !dbg !36
  %0 = load i32, i32* %m1, align 4, !dbg !36, !tbaa !37
  ret i32 %0, !dbg !43
}

; CHECK:        .long   7                       # BTF_KIND_TYPEDEF(id = 4)
; CHECK-NEXT:   .long   134217728               # 0x8000000
; CHECK-NEXT:   .long   5
; CHECK-NEXT:   .long   12                      # BTF_KIND_TYPEDEF(id = 5)
; CHECK-NEXT:   .long   134217728               # 0x8000000
; CHECK-NEXT:   .long   11

; CHECK:        .long   69                      # BTF_KIND_STRUCT(id = 10)
; CHECK-NEXT:   .long   67108865                # 0x4000001
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   72
; CHECK-NEXT:   .long   4
; CHECK-NEXT:   .long   0                       # 0x0

; CHECK:        .long   75                      # BTF_KIND_STRUCT(id = 11)
; CHECK-NEXT:   .long   67108866                # 0x4000002
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   78
; CHECK-NEXT:   .long   7
; CHECK-NEXT:   .long   0                       # 0x0
; CHECK-NEXT:   .long   81
; CHECK-NEXT:   .long   7
; CHECK-NEXT:   .long   32                      # 0x20

; CHECK:        .ascii  "__tt"                  # string offset=7
; CHECK:        .ascii  "_tt"                   # string offset=12
; CHECK:        .ascii  "s2"                    # string offset=69
; CHECK:        .ascii  "m3"                    # string offset=72
; CHECK:        .ascii  "tt"                    # string offset=75
; CHECK:        .ascii  "m1"                    # string offset=78
; CHECK:        .ascii  "m2"                    # string offset=81

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
!7 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 5, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 4, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "mp", scope: !12, file: !1, line: 4, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "__tt", file: !1, line: 3, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "_tt", file: !1, line: 2, baseType: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tt", file: !1, line: 10, size: 64, elements: !19)
!19 = !{!20, !21}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "m1", scope: !18, file: !1, line: 10, baseType: !10, size: 32)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "m2", scope: !18, file: !1, line: 10, baseType: !10, size: 32, offset: 32)
!22 = !{!23}
!23 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 5, type: !11)
!24 = !DILocation(line: 0, scope: !7)
!25 = !DILocation(line: 7, column: 3, scope: !7)
!26 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 12, type: !27, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !33)
!27 = !DISubroutineType(types: !28)
!28 = !{!10, !29}
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !30, size: 64)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s2", file: !1, line: 11, size: 64, elements: !31)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "m3", scope: !30, file: !1, line: 11, baseType: !16, size: 64)
!33 = !{!34}
!34 = !DILocalVariable(name: "arg", arg: 1, scope: !26, file: !1, line: 12, type: !29)
!35 = !DILocation(line: 0, scope: !26)
!36 = !DILocation(line: 14, column: 18, scope: !26)
!37 = !{!38, !40, i64 0}
!38 = !{!"s2", !39, i64 0}
!39 = !{!"tt", !40, i64 0, !40, i64 4}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C/C++ TBAA"}
!43 = !DILocation(line: 14, column: 3, scope: !26)
