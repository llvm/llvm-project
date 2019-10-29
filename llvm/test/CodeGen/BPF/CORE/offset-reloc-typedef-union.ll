; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
;
; Source code:
;   typedef int _int;
;   typedef _int __int;
;   union __s { __int a; __int b; };
;   typedef union __s _s;
;   typedef _s s;
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const void *addr);
;   int test(s *arg) {
;     return get_value(_(&arg->b));
;   }
; clang -target bpf -S -O2 -g -emit-llvm test.c

%union.__s = type { i32 }

; Function Attrs: nounwind
define dso_local i32 @test(%union.__s* %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %union.__s* %arg, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = tail call %union.__s* @llvm.preserve.union.access.index.p0s_union.__ss.p0s_union.__ss(%union.__s* %arg, i32 1), !dbg !23, !llvm.preserve.access.index !14
  %1 = bitcast %union.__s* %0 to i8*, !dbg !23
  %call = tail call i32 @get_value(i8* %1) #4, !dbg !24
  ret i32 %call, !dbg !25
}

; CHECK:        .cfi_startproc
; CHECK: [[RELOC:.Ltmp[0-9]+]]:
; CHECK:         r2 = 0
; CHECK:         r1 += r2
; CHECK:         call get_value

; CHECK:         .long   {{[0-9]+}}              # BTF_KIND_UNION(id = [[TYPE_ID:[0-9]+]])
; CHECK:         .ascii  ".text"                 # string offset=[[SEC_INDEX:[0-9]+]]
; CHECK-NEXT:    .byte   0
; CHECK:         .ascii  "0:1"                   # string offset=[[ACCESS_STR:[0-9]+]]
; CHECK-NEXT:    .byte   0
; CHECK:         .long   16                      # FieldReloc
; CHECK-NEXT:    .long   [[SEC_INDEX]]           # Field reloc section string offset=[[SEC_INDEX]]
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   [[RELOC]]
; CHECK-NEXT:    .long   [[TYPE_ID]]
; CHECK-NEXT:    .long   [[ACCESS_STR]]
; CHECK-NEXT:    .long   0

declare dso_local i32 @get_value(i8*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare %union.__s* @llvm.preserve.union.access.index.p0s_union.__ss.p0s_union.__ss(%union.__s*, i32 immarg) #2

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
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !20)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "s", file: !1, line: 5, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "_s", file: !1, line: 4, baseType: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "__s", file: !1, line: 3, size: 32, elements: !15)
!15 = !{!16, !19}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 3, baseType: !17, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !1, line: 2, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "_int", file: !1, line: 1, baseType: !10)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 3, baseType: !17, size: 32)
!20 = !{!21}
!21 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 8, type: !11)
!22 = !DILocation(line: 0, scope: !7)
!23 = !DILocation(line: 9, column: 20, scope: !7)
!24 = !DILocation(line: 9, column: 10, scope: !7)
!25 = !DILocation(line: 9, column: 3, scope: !7)
