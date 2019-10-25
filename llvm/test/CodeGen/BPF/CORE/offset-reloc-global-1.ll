; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source code:
;   typedef struct v3 { int a; int b; } __v3;
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const int *arg);
;   __v3 g __attribute__((section("stats")));
;   int test() {
;     return get_value(_(&g.b));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.v3 = type { i32, i32 }

@g = dso_local global %struct.v3 zeroinitializer, section "stats", align 4, !dbg !0

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !16 {
entry:
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v3s(%struct.v3* nonnull @g, i32 1, i32 1), !dbg !19, !llvm.preserve.access.index !7
  %call = tail call i32 @get_value(i32* %0) #3, !dbg !20
  ret i32 %call, !dbg !21
}

; CHECK:              r2 = 4
; CHECK:              r1 = g ll
; CHECK:              r1 += r2
; CHECK:              call get_value

; CHECK:              .long   16                      # BTF_KIND_STRUCT(id = [[TID1:[0-9]+]])

; CHECK:              .ascii  ".text"                 # string offset=10
; CHECK:              .ascii  "v3"                    # string offset=16
; CHECK:              .ascii  "0:1"                   # string offset=23

; CHECK:              .long   16                      # FieldReloc
; CHECK-NEXT:         .long   10                      # Field reloc section string offset=10
; CHECK-NEXT:         .long   1
; CHECK-NEXT:         .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:         .long   [[TID1]]
; CHECK-NEXT:         .long   23
; CHECK-NEXT:         .long   0

declare dso_local i32 @get_value(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v3s(%struct.v3*, i32, i32) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v3", file: !3, line: 1, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !3, line: 1, size: 64, elements: !8)
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !3, line: 1, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !7, file: !3, line: 1, baseType: !10, size: 32, offset: 32)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!16 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !17, scopeLine: 5, isDefinition: true, isOptimized: true, unit: !2, retainedNodes: !4)
!17 = !DISubroutineType(types: !18)
!18 = !{!10}
!19 = !DILocation(line: 6, column: 20, scope: !16)
!20 = !DILocation(line: 6, column: 10, scope: !16)
!21 = !DILocation(line: 6, column: 3, scope: !16)
