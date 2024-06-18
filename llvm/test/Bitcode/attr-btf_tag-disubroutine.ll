; RUN: llvm-as < %s | llvm-dis | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   int (__tag1 * g)(void);

; Compilation flag:
;   clang -S -g -emit-llvm test.c

@g = dso_local global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git c15ba1bb9498fa04f6c374337313df43486c9713)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "2ed8742fd12b44b948de1ac5e433bd63")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DISubroutineType(types: !7, annotations: !9)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !{!"btf:type_tag", !"tag1"}

; CHECK: distinct !DIGlobalVariable(name: "g", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L1:[0-9]+]], isLocal: false, isDefinition: true)
; CHECK: ![[L1]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L2:[0-9]+]], size: [[#]])
; CHECK: ![[L2]] = !DISubroutineType(types: ![[#]], annotations: ![[L3:[0-9]+]])
; CHECK: ![[L3]] = !{![[L4:[0-9]+]]}
; CHECK: ![[L4]] = !{!"btf:type_tag", !"tag1"}

!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 7, !"uwtable", i32 2}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git c15ba1bb9498fa04f6c374337313df43486c9713)"}
