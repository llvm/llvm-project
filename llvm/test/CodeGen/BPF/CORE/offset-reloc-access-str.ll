; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
;
; Source code:
;   struct s { int a; int b; };
;   struct t { int c; int d; };
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const void *addr1, const void *addr2);
;   int test(struct s *arg1, struct t *arg2) {
;     return get_value(_(&arg1->b), _(&arg2->d));
;   }
; clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.s = type { i32, i32 }
%struct.t = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.s* %arg1, %struct.t* %arg2) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %struct.s* %arg1, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata %struct.t* %arg2, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s* %arg1, i32 1, i32 1), !dbg !25, !llvm.preserve.access.index !12
  %1 = bitcast i32* %0 to i8*, !dbg !25
  %2 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ts(%struct.t* %arg2, i32 1, i32 1), !dbg !26, !llvm.preserve.access.index !17
  %3 = bitcast i32* %2 to i8*, !dbg !26
  %call = tail call i32 @get_value(i8* %1, i8* %3) #4, !dbg !27
  ret i32 %call, !dbg !28
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK:             .ascii  ".text"                 # string offset=[[SEC_INDEX:[0-9]+]]
; CHECK-NEXT:        .byte   0
; CHECK:             .ascii  "0:1"                   # string offset=[[ACCESS_STR:[0-9]+]]
; CHECK-NEXT:        .byte   0
; CHECK:             .section        .BTF.ext,"",@progbits
; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   [[SEC_INDEX]]           # Field reloc section string offset=[[SEC_INDEX]]
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   [[ACCESS_STR]]
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   [[ACCESS_STR]]
; CHECK-NEXT:        .long   0

declare dso_local i32 @get_value(i8*, i8*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ts(%struct.t*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 366831) (llvm/trunk 366867)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/core-bugs")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (trunk 366831) (llvm/trunk 366867)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !21)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !16}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 64, elements: !13)
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !1, line: 1, baseType: !10, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !12, file: !1, line: 1, baseType: !10, size: 32, offset: 32)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 2, size: 64, elements: !18)
!18 = !{!19, !20}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !17, file: !1, line: 2, baseType: !10, size: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !17, file: !1, line: 2, baseType: !10, size: 32, offset: 32)
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "arg1", arg: 1, scope: !7, file: !1, line: 5, type: !11)
!23 = !DILocalVariable(name: "arg2", arg: 2, scope: !7, file: !1, line: 5, type: !16)
!24 = !DILocation(line: 0, scope: !7)
!25 = !DILocation(line: 6, column: 20, scope: !7)
!26 = !DILocation(line: 6, column: 33, scope: !7)
!27 = !DILocation(line: 6, column: 10, scope: !7)
!28 = !DILocation(line: 6, column: 3, scope: !7)
