; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; From source:
; struct foo {
;   int __attribute__((always_inline)) func(int x) { return x + 2; }
; };

; int i;

; int main() {
;   return foo().func(i);
; }

; CHECK: DW_TAG_structure_type
; CHECK:   DW_TAG_subprogram

; But make sure we emit DW_AT_object_pointer on the abstract definition.
; CHECK: [[ABSTRACT_ORIGIN:.*]]: DW_TAG_subprogram
; CHECK-NOT: {{NULL|TAG}}
; CHECK: DW_AT_specification {{.*}} "_ZN3foo4funcEi"
; CHECK-NOT: {{NULL|TAG}}
; CHECK: DW_AT_object_pointer

; Ensure we omit DW_AT_object_pointer on inlined subroutines.
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} {[[ABSTRACT_ORIGIN]]} "_ZN3foo4funcEi"
; CHECK-NOT: NULL
; CHECK-NOT: DW_AT_object_pointer
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_AT_artificial
; CHECK: DW_TAG

%struct.foo = type { i8 }

@i = global i32 0, align 4

; Function Attrs: uwtable
define i32 @main() #0 !dbg !13 {
entry:
  %this.addr.i = alloca %struct.foo*, align 8
  %x.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  %tmp = alloca %struct.foo, align 1
  store i32 0, i32* %retval
  %0 = load i32, i32* @i, align 4, !dbg !23
  store %struct.foo* %tmp, %struct.foo** %this.addr.i, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr.i, metadata !24, metadata !DIExpression()), !dbg !26
  store i32 %0, i32* %x.addr.i, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr.i, metadata !27, metadata !DIExpression()), !dbg !28
  %this1.i = load %struct.foo*, %struct.foo** %this.addr.i
  %1 = load i32, i32* %x.addr.i, align 4, !dbg !28
  %add.i = add nsw i32 %1, 2, !dbg !28
  ret i32 %add.i, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !12, globals: !18, imports: !2)
!1 = !DIFile(filename: "inline.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS3foo")
!5 = !{!6}
!6 = !DISubprogram(name: "func", linkageName: "_ZN3foo4funcEi", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !"_ZTS3foo", type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS3foo")
!12 = !{!13, !17}
!13 = distinct !DISubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 7, file: !1, scope: !14, type: !15, variables: !2)
!14 = !DIFile(filename: "inline.cpp", directory: "/tmp/dbginfo")
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = distinct !DISubprogram(name: "func", linkageName: "_ZN3foo4funcEi", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !"_ZTS3foo", type: !7, declaration: !6, variables: !2)
!18 = !{!19}
!19 = !DIGlobalVariable(name: "i", line: 5, isLocal: false, isDefinition: true, scope: null, file: !14, type: !9, variable: i32* @i)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 1, !"Debug Info Version", i32 3}
!22 = !{!"clang version 3.5.0 "}
!23 = !DILocation(line: 8, scope: !13)
!24 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !17, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS3foo")
!26 = !DILocation(line: 0, scope: !17, inlinedAt: !23)
!27 = !DILocalVariable(name: "x", line: 2, arg: 2, scope: !17, file: !14, type: !9)
!28 = !DILocation(line: 2, scope: !17, inlinedAt: !23)
