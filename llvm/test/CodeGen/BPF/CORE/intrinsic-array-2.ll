; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source:
;   enum { FIELD_EXISTENCE = 2, };
;   struct s1 { int a1; };
;   int test() {
;     struct s1 *v = 0;
;     return __builtin_preserve_field_info(v[0], FIELD_EXISTENCE);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.s1 = type { i32 }

; Function Attrs: nounwind readnone
define dso_local i32 @test() local_unnamed_addr #0 !dbg !17 {
entry:
  call void @llvm.dbg.value(metadata %struct.s1* null, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = tail call %struct.s1* @llvm.preserve.array.access.index.p0s_struct.s1s.p0s_struct.s1s(%struct.s1* null, i32 0, i32 0), !dbg !23, !llvm.preserve.access.index !8
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0s_struct.s1s(%struct.s1* %0, i64 2), !dbg !24
  ret i32 %1, !dbg !25
}

; CHECK:             .long   16                      # BTF_KIND_STRUCT(id = 4)

; CHECK:             .ascii  ".text"                 # string offset=10
; CHECK:             .ascii  "s1"                    # string offset=16
; CHECK:             .byte   48                      # string offset=22

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   10                      # Field reloc section string offset=10
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   22
; CHECK-NEXT:        .long   2

; Function Attrs: nounwind readnone
declare %struct.s1* @llvm.preserve.array.access.index.p0s_struct.s1s.p0s_struct.s1s(%struct.s1*, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0s_struct.s1s(%struct.s1*, i64 immarg) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 32791937d7aceb0a5e1eaabf1bb1a6dbe1639792)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !7, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 1, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "FIELD_EXISTENCE", value: 2, isUnsigned: true)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 2, size: 32, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !9, file: !1, line: 2, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 32791937d7aceb0a5e1eaabf1bb1a6dbe1639792)"}
!17 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !18, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{!12}
!20 = !{!21}
!21 = !DILocalVariable(name: "v", scope: !17, file: !1, line: 4, type: !8)
!22 = !DILocation(line: 0, scope: !17)
!23 = !DILocation(line: 5, column: 40, scope: !17)
!24 = !DILocation(line: 5, column: 10, scope: !17)
!25 = !DILocation(line: 5, column: 3, scope: !17)
