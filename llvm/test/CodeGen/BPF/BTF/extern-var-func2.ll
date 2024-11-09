; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   extern int elsewhere(void);
;   struct {
;     void *values[];
;   } prog_map = { .values = { elsewhere } };
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c
; ModuleID = 'b.c'

@prog_map = dso_local local_unnamed_addr global { [1 x ptr] } { [1 x ptr] [ptr @elsewhere] }, align 8, !dbg !0

declare !dbg !17 dso_local i32 @elsewhere() #0

; CHECK:             .long   0                               # BTF_KIND_FUNC_PROTO(id = 6)
; CHECK-NEXT:        .long   218103808                       # 0xd000000
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   37                              # BTF_KIND_INT(id = 7)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   41                              # BTF_KIND_FUNC(id = 8)
; CHECK-NEXT:        .long   201326594                       # 0xc000002
; CHECK-NEXT:        .long   6

attributes #0 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "prog_map", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 19.0.0git (git@github.com:llvm/llvm-project.git 0390a6803608e3a5314315b73740c2d3f5a5723f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "b.c", directory: "/home/nickz/llvm-project.git", checksumkind: CSK_MD5, checksum: "41cc17375f1261a0e072590833492553")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 2, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "values", scope: !5, file: !3, line: 3, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, elements: !10)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !{!11}
!11 = !DISubrange(count: -1)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{!"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 0390a6803608e3a5314315b73740c2d3f5a5723f)"}
!17 = !DISubprogram(name: "elsewhere", scope: !3, file: !3, line: 1, type: !18, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!18 = !DISubroutineType(types: !19)
!19 = !{!20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
