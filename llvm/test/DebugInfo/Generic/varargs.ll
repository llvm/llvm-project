; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; REQUIRES: object-emission
;
; Test debug info for variadic function arguments.
; Created from tools/clang/tests/CodeGenCXX/debug-info-varargs.cpp
;
; The ... parameter of variadic should be emitted as
; DW_TAG_unspecified_parameters.
;
; Normal variadic function.
; void b(int c, ...);
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "a"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "b"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; Variadic C++ member function.
; struct A { void a(int c, ...); }
;
; Variadic function pointer.
; void (*fptr)(int, ...);
;
; CHECK: DW_TAG_subroutine_type
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; ModuleID = 'llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp'

%struct.A = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1biz(i32 %c, ...) #0 !dbg !14 {
  %1 = alloca i32, align 4
  %a = alloca %struct.A, align 1
  %fptr = alloca void (i32, ...)*, align 8
  store i32 %c, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata void (i32, ...)** %fptr, metadata !25, metadata !DIExpression()), !dbg !27
  store void (i32, ...)* @_Z1biz, void (i32, ...)** %fptr, align 8, !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !3, subprograms: !13, globals: !2, imports: !2)
!1 = !DIFile(filename: "llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp", directory: "radar/13690847")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 3, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6}
!6 = !DISubprogram(name: "a", linkageName: "_ZN1A1aEiz", line: 6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !"_ZTS1A", type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !10, null}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1A")
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = distinct !DISubprogram(name: "b", linkageName: "_Z1biz", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 13, file: !1, scope: !15, type: !16, variables: !2)
!15 = !DIFile(filename: "llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp", directory: "radar/13690847")
!16 = !DISubroutineType(types: !17)
!17 = !{null, !10, null}
!18 = !{i32 2, !"Dwarf Version", i32 2}
!19 = !{i32 1, !"Debug Info Version", i32 3}
!20 = !{!"clang version 3.5 "}
!21 = !DILocalVariable(name: "c", line: 13, arg: 1, scope: !14, file: !15, type: !10)
!22 = !DILocation(line: 13, scope: !14)
!23 = !DILocalVariable(name: "a", line: 16, scope: !14, file: !15, type: !4)
!24 = !DILocation(line: 16, scope: !14)
!25 = !DILocalVariable(name: "fptr", line: 18, scope: !14, file: !15, type: !26)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !16)
!27 = !DILocation(line: 18, scope: !14)
!28 = !DILocation(line: 22, scope: !14)
