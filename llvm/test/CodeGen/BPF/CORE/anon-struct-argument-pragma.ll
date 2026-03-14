; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -o - %t1 | FileCheck %s
;
; Source:
;   #pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
;   typedef struct {
;       union {
;               void	*kernel;
;               void	*user;
;       };
;       unsigned is_kernel : 1;
;   } sockptr_t;
;   #pragma clang attribute pop
;   int test(sockptr_t *arg) {
;     return arg->is_kernel;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -g -Xclang -disable-llvm-passes test.c

%struct.sockptr_t = type { %union.anon, i8 }
%union.anon = type { ptr }

; Function Attrs: nounwind
define dso_local i32 @test(ptr noundef %arg) #0 !dbg !7 {
entry:
  %arg.addr = alloca ptr, align 8
  store ptr %arg, ptr %arg.addr, align 8, !tbaa !25
  call void @llvm.dbg.declare(metadata ptr %arg.addr, metadata !24, metadata !DIExpression()), !dbg !29
  %0 = load ptr, ptr %arg.addr, align 8, !dbg !30, !tbaa !25
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.sockptr_t) %0, i32 1, i32 1), !dbg !31, !llvm.preserve.access.index !13
  %bf.load = load i8, ptr %1, align 8, !dbg !31
  %bf.clear = and i8 %bf.load, 1, !dbg !31
  %bf.cast = zext i8 %bf.clear to i32, !dbg !31
  ret i32 %bf.cast, !dbg !32
}

; CHECK:             .long   1                               # BTF_KIND_TYPEDEF(id = 2)
; CHECK-NEXT:        .long   134217728                       # 0x8000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                               # BTF_KIND_STRUCT(id = 3)

; CHECK:             .ascii  "sockptr_t"                     # string offset=1
; CHECK:             .ascii  ".text"                         # string offset=59
; CHECK:             .ascii  "0:1"                           # string offset=65

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   59                              # Field reloc section string offset=59
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp[[#]]
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   65
; CHECK-NEXT:        .long   0

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind readnone willreturn
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nocallback nofree nosync nounwind readnone willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git d81a8759c969344c1e96992aab30f5b5a9d5ffd3)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/test/anon", checksumkind: CSK_MD5, checksum: "7ba33bf2146cc86b1c8396f6d3eace81")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git d81a8759c969344c1e96992aab30f5b5a9d5ffd3)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "sockptr_t", file: !1, line: 8, baseType: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 2, size: 128, elements: !14)
!14 = !{!15, !21}
!15 = !DIDerivedType(tag: DW_TAG_member, scope: !13, file: !1, line: 3, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !13, file: !1, line: 3, size: 64, elements: !17)
!17 = !{!18, !20}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "kernel", scope: !16, file: !1, line: 4, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "user", scope: !16, file: !1, line: 5, baseType: !19, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "is_kernel", scope: !13, file: !1, line: 7, baseType: !22, size: 1, offset: 64, flags: DIFlagBitField, extraData: i64 64)
!22 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!23 = !{!24}
!24 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 10, type: !11)
!25 = !{!26, !26, i64 0}
!26 = !{!"any pointer", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 10, column: 21, scope: !7)
!30 = !DILocation(line: 11, column: 10, scope: !7)
!31 = !DILocation(line: 11, column: 15, scope: !7)
!32 = !DILocation(line: 11, column: 3, scope: !7)
