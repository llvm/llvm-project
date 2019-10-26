; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
;
; Source Code:
;   #define _(x) (__builtin_preserve_access_index(x))
;   struct s {int a; int b;};
;   const void *test(struct s *arg) { return _(&arg->b); }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm test.c

%struct.s = type { i32, i32 }

; Function Attrs: nounwind readnone
define dso_local i8* @test(%struct.s* readnone %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %struct.s* %arg, metadata !19, metadata !DIExpression()), !dbg !20
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s* %arg, i32 1, i32 1), !dbg !21, !llvm.preserve.access.index !13
  %1 = bitcast i32* %0 to i8*, !dbg !21
  ret i8* %1, !dbg !22
}

; CHECK-LABEL: test
; CHECK:       r0 = r1
; CHECK:       r1 = 4
; CHECK:       r0 += r1
; CHECK:       exit
;
; CHECK:       .long   1                       # BTF_KIND_STRUCT(id = 2)
;
; CHECK:       .byte   115                     # string offset=1
; CHECK:       .ascii  ".text"                 # string offset=20
; CHECK:       .ascii  "0:1"                   # string offset=63
;
; CHECK:       .long   16                      # FieldReloc
; CHECK-NEXT:  .long   20                      # Field reloc section string offset=20
; CHECK-NEXT:  .long   1
; CHECK-NEXT:  .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:  .long   2
; CHECK-NEXT:  .long   63
; CHECK-NEXT:  .long   0

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s*, i32, i32) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6e353b4df3aa452ed4741a5e5caea02b1a876d8c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6e353b4df3aa452ed4741a5e5caea02b1a876d8c)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !18)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 2, size: 64, elements: !14)
!14 = !{!15, !17}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 2, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !1, line: 2, baseType: !16, size: 32, offset: 32)
!18 = !{!19}
!19 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 3, type: !12)
!20 = !DILocation(line: 0, scope: !7)
!21 = !DILocation(line: 3, column: 42, scope: !7)
!22 = !DILocation(line: 3, column: 35, scope: !7)
