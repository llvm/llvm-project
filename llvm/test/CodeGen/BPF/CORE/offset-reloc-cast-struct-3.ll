; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct v1 { int a; int b; };
;   typedef struct v1 __v1;
;   typedef int __int;
;   struct v3 { char c; __int d[40]; };
;   typedef struct v3 __v3;
;   #define _(x) (__builtin_preserve_access_index(x))
;   #define cast_to_v1(x) ((__v1 *)(x))
;   int get_value(const int *arg);
;   int test(__v3 *arg) {
;     return get_value(_(&cast_to_v1(&arg->d[4])->b));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.v3 = type { i8, [40 x i32] }
%struct.v1 = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(ptr %arg) local_unnamed_addr #0 !dbg !19 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !30, metadata !DIExpression()), !dbg !31
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.v3) %arg, i32 1, i32 1), !dbg !32, !llvm.preserve.access.index !24
  %1 = tail call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([40 x i32]) %0, i32 1, i32 4), !dbg !32, !llvm.preserve.access.index !11
  %2 = bitcast ptr %1 to ptr, !dbg !32
  %3 = tail call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.v1) %2, i32 1, i32 1), !dbg !32, !llvm.preserve.access.index !6
  %call = tail call i32 @get_value(ptr %3) #4, !dbg !33
  ret i32 %call, !dbg !34
}

; CHECK:             r2 = 20
; CHECK:             r1 += r2
; CHECK:             r2 = 4
; CHECK:             r1 += r2
; CHECK:             call get_value

; CHECK:             .long   6                       # BTF_KIND_STRUCT(id = [[TID1:[0-9]+]])
; CHECK:             .long   111                     # BTF_KIND_STRUCT(id = [[TID2:[0-9]+]])

; CHECK:             .ascii  "v3"                    # string offset=6
; CHECK:             .ascii  ".text"                 # string offset=57
; CHECK:             .ascii  "0:1:4"                 # string offset=63
; CHECK:             .ascii  "v1"                    # string offset=111
; CHECK:             .ascii  "0:1"                   # string offset=118

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   57                      # Field reloc section string offset=57
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[TID1]]
; CHECK-NEXT:        .long   63
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[TID2]]
; CHECK-NEXT:        .long   118
; CHECK-NEXT:        .long   0

declare dso_local i32 @get_value(ptr) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32, i32) #2

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32, i32) #2

; Function Attrs: nounwind readnone

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4, !11}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v1", file: !1, line: 2, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v1", file: !1, line: 1, size: 64, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !1, line: 1, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !1, line: 1, baseType: !9, size: 32, offset: 32)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 1280, elements: !13)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !1, line: 3, baseType: !9)
!13 = !{!14}
!14 = !DISubrange(count: 40)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!19 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 9, type: !20, scopeLine: 9, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !29)
!20 = !DISubroutineType(types: !21)
!21 = !{!9, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v3", file: !1, line: 5, baseType: !24)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !1, line: 4, size: 1312, elements: !25)
!25 = !{!26, !28}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !24, file: !1, line: 4, baseType: !27, size: 8)
!27 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !24, file: !1, line: 4, baseType: !11, size: 1280, offset: 32)
!29 = !{!30}
!30 = !DILocalVariable(name: "arg", arg: 1, scope: !19, file: !1, line: 9, type: !22)
!31 = !DILocation(line: 0, scope: !19)
!32 = !DILocation(line: 10, column: 20, scope: !19)
!33 = !DILocation(line: 10, column: 10, scope: !19)
!34 = !DILocation(line: 10, column: 3, scope: !19)
