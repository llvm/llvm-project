; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -o - %t1 | FileCheck %s
;
; Source:
;   #pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
;   typedef union {
;       struct {
;               void	*kernel;
;               void	*user;
;       };
;       unsigned is_kernel : 1;
;   } sockptr_t;
;   #pragma clang attribute pop
;   void *foo(void);
;   int test() {
;     sockptr_t *arg = foo();
;     return __builtin_preserve_field_info(arg->is_kernel, 1);
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -g -Xclang -disable-llvm-passes test.c

%union.sockptr_t = type { %struct.anon }
%struct.anon = type { ptr, ptr }

; Function Attrs: nounwind
define dso_local i32 @test() #0 !dbg !7 {
entry:
  %arg = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr %arg) #6, !dbg !25
  call void @llvm.dbg.declare(metadata ptr %arg, metadata !12, metadata !DIExpression()), !dbg !26
  %call = call ptr @foo(), !dbg !27
  store ptr %call, ptr %arg, align 8, !dbg !26, !tbaa !28
  %0 = load ptr, ptr %arg, align 8, !dbg !32, !tbaa !28
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%union.sockptr_t) %0, i32 0, i32 1), !dbg !33, !llvm.preserve.access.index !15
  %2 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %1, i64 1), !dbg !34
  call void @llvm.lifetime.end.p0(i64 8, ptr %arg) #6, !dbg !35
  ret i32 %2, !dbg !36
}

; CHECK:             .long   56                              # BTF_KIND_TYPEDEF(id = 7)
; CHECK-NEXT:        .long   134217728                       # 0x8000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   0                               # BTF_KIND_UNION(id = 8)

; CHECK:             .ascii  ".text"                         # string offset=10
; CHECK:             .ascii  "sockptr_t"                     # string offset=56
; CHECK:             .ascii  "0:1"                           # string offset=101


; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   10                              # Field reloc section string offset=10
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp[[#]]
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   101
; CHECK-NEXT:        .long   1

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare !dbg !37 dso_local ptr @foo() #3

; Function Attrs: nocallback nofree nosync nounwind readnone willreturn
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #4

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg) #5

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nocallback nofree nosync nounwind readnone willreturn }
attributes #5 = { nounwind readnone }
attributes #6 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 8c7d5118961e7ffc0304126ec2122d21e2eb1f79)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/test/anon", checksumkind: CSK_MD5, checksum: "2c5f698241a8b5ddf345a5743dfca258")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 8c7d5118961e7ffc0304126ec2122d21e2eb1f79)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "arg", scope: !7, file: !1, line: 12, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "sockptr_t", file: !1, line: 8, baseType: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_union_type, file: !1, line: 2, size: 128, elements: !16)
!16 = !{!17, !23}
!17 = !DIDerivedType(tag: DW_TAG_member, scope: !15, file: !1, line: 3, baseType: !18, size: 128)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !15, file: !1, line: 3, size: 128, elements: !19)
!19 = !{!20, !22}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "kernel", scope: !18, file: !1, line: 4, baseType: !21, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "user", scope: !18, file: !1, line: 5, baseType: !21, size: 64, offset: 64)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "is_kernel", scope: !15, file: !1, line: 7, baseType: !24, size: 1, flags: DIFlagBitField, extraData: i64 0)
!24 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!25 = !DILocation(line: 12, column: 3, scope: !7)
!26 = !DILocation(line: 12, column: 14, scope: !7)
!27 = !DILocation(line: 12, column: 20, scope: !7)
!28 = !{!29, !29, i64 0}
!29 = !{!"any pointer", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !DILocation(line: 13, column: 40, scope: !7)
!33 = !DILocation(line: 13, column: 45, scope: !7)
!34 = !DILocation(line: 13, column: 10, scope: !7)
!35 = !DILocation(line: 14, column: 1, scope: !7)
!36 = !DILocation(line: 13, column: 3, scope: !7)
!37 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 10, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !40)
!38 = !DISubroutineType(types: !39)
!39 = !{!21}
!40 = !{}
