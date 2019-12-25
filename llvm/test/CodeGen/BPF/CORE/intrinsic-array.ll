; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
;
; Source code:
;   #define _(x) (__builtin_preserve_access_index(x))
;   struct s { int a; int b; };
;   int get_value(const void *addr);
;   int test(struct s *arg) { return get_value(_(&arg[2].b)); }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm test.c

%struct.s = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.s* %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %struct.s* %arg, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = tail call %struct.s* @llvm.preserve.array.access.index.p0s_struct.ss.p0s_struct.ss(%struct.s* %arg, i32 0, i32 2), !dbg !19, !llvm.preserve.access.index !11
  %1 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s* %0, i32 1, i32 1), !dbg !19, !llvm.preserve.access.index !12
  %2 = bitcast i32* %1 to i8*, !dbg !19
  %call = tail call i32 @get_value(i8* %2) #4, !dbg !20
  ret i32 %call, !dbg !21
}
; CHECK-LABEL: test
; CHECK:       [[RELOC:.Ltmp[0-9]+]]
; CHECK:       r2 = 20
; CHECK:       r1 += r2
; CHECK:       call get_value
; CHECK:       exit
;
; CHECK:      .section        .BTF.ext,"",@progbits
; CHECK:      .long   16                      # FieldReloc
; CHECK-NEXT: .long   20                      # Field reloc section string offset=20
; CHECK-NEXT: .long   1
; CHECK-NEXT: .long   [[RELOC]]
; CHECK-NEXT: .long   2
; CHECK-NEXT: .long   26
; CHECK-NEXT: .long   0

declare dso_local i32 @get_value(i8*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare %struct.s* @llvm.preserve.array.access.index.p0s_struct.ss.p0s_struct.ss(%struct.s*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 365789)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 365789)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 2, size: 64, elements: !13)
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !1, line: 2, baseType: !10, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !12, file: !1, line: 2, baseType: !10, size: 32, offset: 32)
!16 = !{!17}
!17 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 4, type: !11)
!18 = !DILocation(line: 0, scope: !7)
!19 = !DILocation(line: 4, column: 44, scope: !7)
!20 = !DILocation(line: 4, column: 34, scope: !7)
!21 = !DILocation(line: 4, column: 27, scope: !7)
