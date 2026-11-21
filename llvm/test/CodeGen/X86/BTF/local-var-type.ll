; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck %s

; Verify dbg.declare-only local variable types are emitted on x86_64.
;
; Source:
;   int f1(void) {
;     struct local_ctx { int fd; } ctx = { .fd = 7 };
;     return ctx.fd;
;   }
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -gbtf -S -emit-llvm t.c

%struct.local_ctx = type { i32 }

define dso_local i32 @f1() !dbg !7 {
entry:
  %ctx = alloca %struct.local_ctx, align 4
  call void @llvm.dbg.declare(metadata ptr %ctx, metadata !12, metadata !DIExpression()), !dbg !17
  store i32 7, ptr %ctx, align 4, !dbg !17
  %fd = load i32, ptr %ctx, align 4, !dbg !18
  ret i32 %fd, !dbg !19
}

; CHECK: .section        .BTF,"",@progbits
; CHECK: # BTF_KIND_STRUCT(
; CHECK: .ascii  "local_ctx"
; CHECK: .ascii  "fd"

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 4, !"BTF", i32 1}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocalVariable(name: "ctx", scope: !7, file: !1, line: 2, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "local_ctx", scope: !7, file: !1, line: 2, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "fd", scope: !13, file: !1, line: 2, baseType: !10, size: 32)
!16 = !DILocation(line: 1, column: 1, scope: !7)
!17 = !DILocation(line: 2, column: 30, scope: !7)
!18 = !DILocation(line: 3, column: 3, scope: !7)
!19 = !DILocation(line: 3, column: 10, scope: !7)
