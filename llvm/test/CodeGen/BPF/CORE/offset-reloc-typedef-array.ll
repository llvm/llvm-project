; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
;
; Source code:
;   typedef const int arr_t[7];
;   typedef arr_t __arr;
;   typedef __arr _arr;
;   struct __s { _arr a; };
;   typedef struct __s s;
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const void *addr);
;   int test(s *arg) {
;     return get_value(_(&arg->a[1]));
;   }
; clang -target bpf -S -O2 -g -emit-llvm test.c

%struct.__s = type { [7 x i32] }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.__s* %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %struct.__s* %arg, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = tail call [7 x i32]* @llvm.preserve.struct.access.index.p0a7i32.p0s_struct.__ss(%struct.__s* %arg, i32 0, i32 0), !dbg !26, !llvm.preserve.access.index !13
  %1 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0a7i32([7 x i32]* %0, i32 1, i32 1), !dbg !26, !llvm.preserve.access.index !19
  %2 = bitcast i32* %1 to i8*, !dbg !26
  %call = tail call i32 @get_value(i8* %2) #4, !dbg !27
  ret i32 %call, !dbg !28
}

; CHECK:        .cfi_startproc
; CHECK: [[RELOC:.Ltmp[0-9]+]]:
; CHECK:         r2 = 4
; CHECK:         r1 += r2
; CHECK:         call get_value

; CHECK:         .long   {{[0-9]+}}              # BTF_KIND_STRUCT(id = [[TYPE_ID:[0-9]+]])
; CHECK:         .ascii  ".text"                 # string offset=[[SEC_INDEX:[0-9]+]]
; CHECK-NEXT:    .byte   0
; CHECK:         .ascii  "0:0:1"                 # string offset=[[ACCESS_STR:[0-9]+]]
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
declare [7 x i32]* @llvm.preserve.struct.access.index.p0a7i32.p0s_struct.__ss(%struct.__s*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0a7i32([7 x i32]*, i32 immarg, i32 immarg) #2

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
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !23)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "s", file: !1, line: 5, baseType: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__s", file: !1, line: 4, size: 224, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 4, baseType: !16, size: 224)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "_arr", file: !1, line: 3, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "__arr", file: !1, line: 2, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "arr_t", file: !1, line: 1, baseType: !19)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 224, elements: !21)
!20 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!21 = !{!22}
!22 = !DISubrange(count: 7)
!23 = !{!24}
!24 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 8, type: !11)
!25 = !DILocation(line: 0, scope: !7)
!26 = !DILocation(line: 9, column: 20, scope: !7)
!27 = !DILocation(line: 9, column: 10, scope: !7)
!28 = !DILocation(line: 9, column: 3, scope: !7)
