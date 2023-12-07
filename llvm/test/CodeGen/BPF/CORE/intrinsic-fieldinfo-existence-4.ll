; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
; Source code:
;   #define BPF_FIELD_EXISTS 2
;   unsigned test1() {
;     struct t {
;       int val;
;     } bar;
;     return __builtin_preserve_field_info((&bar)[1], BPF_FIELD_EXISTS);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

%struct.t = type { i32 }

; Function Attrs: nounwind
define dso_local i32 @test1() #0 !dbg !6 {
entry:
  %bar = alloca %struct.t, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %bar) #5, !dbg !20
  call void @llvm.dbg.declare(metadata ptr %bar, metadata !11, metadata !DIExpression()), !dbg !21
  %0 = call ptr @llvm.preserve.array.access.index.p0.ts.p0.ts(ptr elementtype(%struct.t) %bar, i32 0, i32 1), !dbg !22, !llvm.preserve.access.index !4
  %1 = call i32 @llvm.bpf.preserve.field.info.p0.ts(ptr %0, i64 2), !dbg !23
  call void @llvm.lifetime.end.p0(i64 4, ptr %bar) #5, !dbg !24
  ret i32 %1, !dbg !25
}

; CHECK:             r0 = 1
; CHECK-NEXT:        exit

; CHECK:             .long   26                              # BTF_KIND_STRUCT(id = 4)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   32                              # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020

; CHECK:             .byte   116                             # string offset=26
; CHECK:             .ascii  "val"                           # string offset=28
; CHECK:             .ascii  "int"                           # string offset=32
; CHECK:             .byte   49                              # string offset=36

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   {{[0-9]+}}                      # Field reloc section string
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nofree nosync nounwind readnone willreturn
declare ptr @llvm.preserve.array.access.index.p0.ts.p0.ts(ptr, i32 immarg, i32 immarg) #3

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0.ts(ptr, i64 immarg) #4

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nofree nosync nounwind readnone willreturn }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 2e0ee68dc85c0a2b7e65e489a60ab363393b06a8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", scope: !6, file: !1, line: 3, size: 32, elements: !12)
!6 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !DILocalVariable(name: "bar", scope: !6, file: !1, line: 5, type: !5)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !5, file: !1, line: 4, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{i32 7, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"frame-pointer", i32 2}
!19 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 2e0ee68dc85c0a2b7e65e489a60ab363393b06a8)"}
!20 = !DILocation(line: 3, column: 3, scope: !6)
!21 = !DILocation(line: 5, column: 5, scope: !6)
!22 = !DILocation(line: 6, column: 40, scope: !6)
!23 = !DILocation(line: 6, column: 10, scope: !6)
!24 = !DILocation(line: 7, column: 1, scope: !6)
!25 = !DILocation(line: 6, column: 3, scope: !6)
