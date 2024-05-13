; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   typedef struct v3 { int a; int b; } __v3;
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const int *arg);
;   __v3 g[4][5] __attribute__((section("stats")));
;   int test() {
;     return get_value(_(&g[1][2].b));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.v3 = type { i32, i32 }

@g = dso_local global [4 x [5 x %struct.v3]] zeroinitializer, section "stats", align 4, !dbg !0

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !23 {
entry:
  %0 = tail call ptr @llvm.preserve.array.access.index.p0.v3s.p0.v3s(ptr elementtype([4 x [5 x %struct.v3]]) nonnull @g, i32 1, i32 1), !dbg !26, !llvm.preserve.access.index !6
  %1 = tail call ptr @llvm.preserve.array.access.index.p0.v3s.p0.v3s(ptr elementtype([5 x %struct.v3]) %0, i32 1, i32 2), !dbg !26, !llvm.preserve.access.index !16
  %2 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.v3s(ptr elementtype(%struct.v3) %1, i32 1, i32 1), !dbg !26, !llvm.preserve.access.index !8
  %call = tail call i32 @get_value(ptr %2) #3, !dbg !27
  ret i32 %call, !dbg !28
}

; CHECK:              r2 = 60
; CHECK:              r1 = g ll
; CHECK:              r1 += r2
; CHECK:              call get_value

; CHECK:              .long   16                      # BTF_KIND_STRUCT(id = [[TID1:[0-9]+]])

; CHECK:              .ascii  ".text"                 # string offset=10
; CHECK:              .ascii  "v3"                    # string offset=16
; CHECK:              .ascii  "7:1"                   # string offset=23

; CHECK:              .long   16                      # FieldReloc
; CHECK-NEXT:         .long   10                      # Field reloc section string offset=10
; CHECK-NEXT:         .long   1
; CHECK-NEXT:         .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:         .long   [[TID1]]
; CHECK-NEXT:         .long   23
; CHECK-NEXT:         .long   0


declare dso_local i32 @get_value(ptr) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.array.access.index.p0.v3s.p0.v3s(ptr, i32, i32) #2

; Function Attrs: nounwind readnone

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.v3s(ptr, i32, i32) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19, !20, !21}
!llvm.ident = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !18, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!4 = !{}
!5 = !{!6, !16}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 1280, elements: !13)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v3", file: !3, line: 1, baseType: !8)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !3, line: 1, size: 64, elements: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !8, file: !3, line: 1, baseType: !11, size: 32)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !3, line: 1, baseType: !11, size: 32, offset: 32)
!13 = !{!14, !15}
!14 = !DISubrange(count: 4)
!15 = !DISubrange(count: 5)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 320, elements: !17)
!17 = !{!15}
!18 = !{!0}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!23 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !24, scopeLine: 5, isDefinition: true, isOptimized: true, unit: !2, retainedNodes: !4)
!24 = !DISubroutineType(types: !25)
!25 = !{!11}
!26 = !DILocation(line: 6, column: 20, scope: !23)
!27 = !DILocation(line: 6, column: 10, scope: !23)
!28 = !DILocation(line: 6, column: 3, scope: !23)
