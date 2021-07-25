; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   typedef struct s1 { int a1; int a2:4; int a3;} __s1;
;   enum { FIELD_BYTE_SIZE = 1, };
;   int test(__s1 *arg) {
;     return __builtin_preserve_field_info(arg->a2, FIELD_BYTE_SIZE);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.s1 = type { i32, i8, i32 }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%struct.s1* readnone %arg) local_unnamed_addr #0 !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata %struct.s1* %arg, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = tail call i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %arg, i32 1, i32 1), !dbg !25, !llvm.preserve.access.index !17
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %0, i64 1), !dbg !26
  ret i32 %1, !dbg !27
}

; CHECK:             r0 = 4
; CHECK:             exit

; CHECK:             .long   6                       # BTF_KIND_STRUCT(id = 3)

; CHECK:             .ascii  "s1"                    # string offset=6
; CHECK:             .ascii  ".text"                 # string offset=31
; CHECK:             .ascii  "0:1"                   # string offset=37

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   31                      # Field reloc section string offset=31
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   37
; CHECK-NEXT:        .long   1

; Function Attrs: nounwind readnone
declare i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i8(i8*, i64) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6c63b5d6a1cb84a75807804337c855a41c976260)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 2, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "FIELD_BYTE_SIZE", value: 1, isUnsigned: true)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6c63b5d6a1cb84a75807804337c855a41c976260)"}
!11 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !22)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s1", file: !1, line: 1, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 1, size: 96, elements: !18)
!18 = !{!19, !20, !21}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !17, file: !1, line: 1, baseType: !14, size: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !17, file: !1, line: 1, baseType: !14, size: 4, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "a3", scope: !17, file: !1, line: 1, baseType: !14, size: 32, offset: 64)
!22 = !{!23}
!23 = !DILocalVariable(name: "arg", arg: 1, scope: !11, file: !1, line: 3, type: !15)
!24 = !DILocation(line: 0, scope: !11)
!25 = !DILocation(line: 4, column: 45, scope: !11)
!26 = !DILocation(line: 4, column: 10, scope: !11)
!27 = !DILocation(line: 4, column: 3, scope: !11)
