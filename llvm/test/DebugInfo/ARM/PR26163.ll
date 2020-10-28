; RUN: llc -filetype=obj -o - < %s | llvm-dwarfdump -debug-info - | FileCheck %s
;
; Checks that we're omitting the first range, as it is empty, and that we're
; emitting one that spans the rest of the function. In this case, the first
; range, which we omit, describes 8 bytes of the variable using DW_OP_litX,
; whereas the second one only describes 4 bytes, so clobbering the whole 8 byte
; fragment with the 4 bytes fragment isn't necessarily best thing to do here,
; but it is what is currently being emitted. Any change here needs to be
; intentional, so the test is very specific.
;
; The variable is given a single location instead of a location list entry
; because the function validThroughout has a special code path for single
; locations with a constant value that start in the prologue.
;
; CHECK:      DW_TAG_inlined_subroutine
; CHECK:        DW_TAG_variable
; CHECK-NEXT:     DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_piece 0x4)
; CHECK-NEXT      DW_AT_name ("i4")

; Created form the following test case (PR26163) with
; clang -cc1 -triple armv4t--freebsd11.0-gnueabi -emit-obj -debug-info-kind=standalone -O2 -x c test.c
;
; typedef	unsigned int	size_t;
; struct timeval {
; 	long long tv_sec;
; 	int tv_usec;
; };
;
; void *memset(void *, int, size_t);
; void foo(void);
;
; static void
; bar(int value)
; {
; 	struct timeval lifetime;
;
; 	memset(&lifetime, 0, sizeof(struct timeval));
; 	lifetime.tv_sec = value;
;
; 	foo();
; }
;
; int
; parse_config_file(void)
; {
; 	int value;
;
; 	bar(value);
; 	return (0);
; }

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t--freebsd11.0-gnueabi"

%struct.timeval = type { i64, i32 }

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

declare void @foo()

define i32 @parse_config_file() !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !26), !dbg !27
  tail call void @llvm.dbg.declare(metadata %struct.timeval* undef, metadata !16, metadata !26), !dbg !29
  tail call void @llvm.dbg.value(metadata i64 0, metadata !16, metadata !30), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !31), !dbg !29
  tail call void @foo() #3, !dbg !32
  ret i32 0, !dbg !33
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23, !24}
!llvm.ident = !{!25}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/home/ubuntu/bugs")
!2 = !{}
!4 = distinct !DISubprogram(name: "parse_config_file", scope: !5, file: !5, line: 22, type: !6, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!5 = !DIFile(filename: "test.c", directory: "/home/ubuntu/bugs")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "value", scope: !4, file: !5, line: 24, type: !8)
!11 = distinct !DISubprogram(name: "bar", scope: !5, file: !5, line: 11, type: !12, isLocal: true, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !8}
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "value", arg: 1, scope: !11, file: !5, line: 11, type: !8)
!16 = !DILocalVariable(name: "lifetime", scope: !11, file: !5, line: 13, type: !17)
!17 = !DICompositeType(tag: DW_TAG_structure_type, name: "timeval", file: !5, line: 2, size: 128, align: 64, elements: !18)
!18 = !{!19, !21}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "tv_sec", scope: !17, file: !5, line: 3, baseType: !20, size: 64, align: 64)
!20 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "tv_usec", scope: !17, file: !5, line: 4, baseType: !8, size: 32, align: 32, offset: 64)
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 4}
!24 = !{i32 1, !"min_enum_size", i32 4}
!25 = !{!"clang version 3.9.0"}
!26 = !DIExpression()
!27 = !DILocation(line: 11, scope: !11, inlinedAt: !28)
!28 = distinct !DILocation(line: 26, scope: !4)
!29 = !DILocation(line: 13, scope: !11, inlinedAt: !28)
!30 = !DIExpression(DW_OP_LLVM_fragment, 0, 64)
!31 = !DIExpression(DW_OP_LLVM_fragment, 0, 32)
!32 = !DILocation(line: 18, scope: !11, inlinedAt: !28)
!33 = !DILocation(line: 27, scope: !4)
