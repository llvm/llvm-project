; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Testing that two distinct (distinct by writing them in separate files, while
; still fulfilling C++'s ODR by having identical token sequences) functions,
; linked under LTO, get plausible debug info (and don't crash).

; Built from source:
; $ clang++ a.cpp b.cpp -g -c -emit-llvm
; $ llvm-link a.bc b.bc -o ab.bc

; This change is intended to tickle a case where the subprogram MDNode
; associated with the llvm::Function will differ from the subprogram
; referenced by the DbgLocs in the function.

; $ sed -ie "s/!12, !0/!0, !12/" ab.ll
; $ cat a.cpp
; inline int func(int i) {
;   return i * 2;
; }
; int (*x)(int) = &func;
; $ cat b.cpp
; inline int func(int i) {
;   return i * 2;
; }
; int (*y)(int) = &func;

; The DISubprogram should show up in compile unit a.
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}}"b.cpp"
; CHECK-NOT: DW_TAG_subprogram

; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}}"a.cpp"
; CHECK:     DW_AT_name {{.*}} "func"

@x = global i32 (i32)* @_Z4funci, align 8
@y = global i32 (i32)* @_Z4funci, align 8

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr i32 @_Z4funci(i32 %i) #0 !dbg !4 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !22, metadata !DIExpression()), !dbg !23
  %2 = load i32, i32* %1, align 4, !dbg !24
  %3 = mul nsw i32 %2, 2, !dbg !24
  ret i32 %3, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { inlinehint nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!12, !0}
!llvm.module.flags = !{!19, !20}
!llvm.ident = !{!21, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !9, imports: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "func", linkageName: "_Z4funci", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DIGlobalVariable(name: "x", line: 4, isLocal: false, isDefinition: true, scope: null, file: !5, type: !11, variable: i32 (i32)** @x)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !6)
!12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !13, enums: !2, retainedTypes: !2, globals: !17, imports: !2)
!13 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo")
!15 = distinct !DISubprogram(name: "func", linkageName: "_Z4funci", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !12, scopeLine: 1, file: !13, scope: !16, type: !6, variables: !2)
!16 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo")
!17 = !{!18}
!18 = !DIGlobalVariable(name: "y", line: 4, isLocal: false, isDefinition: true, scope: null, file: !16, type: !11, variable: i32 (i32)** @y)
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 1, !"Debug Info Version", i32 3}
!21 = !{!"clang version 3.5.0 "}
!22 = !DILocalVariable(name: "i", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!23 = !DILocation(line: 1, scope: !4)
!24 = !DILocation(line: 2, scope: !4)
