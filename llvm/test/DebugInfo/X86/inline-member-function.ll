
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

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

source_filename = "test/DebugInfo/X86/inline-member-function.ll"

%struct.foo = type { i8 }

@i = global i32 0, align 4, !dbg !0

; Function Attrs: uwtable
define i32 @main() #0 !dbg !17 {
entry:
  %this.addr.i = alloca %struct.foo*, align 8
  %x.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  %tmp = alloca %struct.foo, align 1
  store i32 0, i32* %retval
  %0 = load i32, i32* @i, align 4, !dbg !20
  store %struct.foo* %tmp, %struct.foo** %this.addr.i, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr.i, metadata !21, metadata !24), !dbg !25
  store i32 %0, i32* %x.addr.i, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr.i, metadata !26, metadata !24), !dbg !27
  %this1.i = load %struct.foo*, %struct.foo** %this.addr.i
  %1 = load i32, i32* %x.addr.i, align 4, !dbg !27
  %add.i = add nsw i32 %1, 2, !dbg !27
  ret i32 %add.i, !dbg !20
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "i", scope: null, file: !2, line: 5, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "inline.cpp", directory: "/tmp/dbginfo")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !6, globals: !13, imports: !5)
!5 = !{}
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !2, line: 1, size: 8, align: 8, elements: !8, identifier: "_ZTS3foo")
!8 = !{!9}
!9 = !DISubprogram(name: "func", linkageName: "_ZN3foo4funcEi", scope: !7, file: !2, line: 2, type: !10, isLocal: false, isDefinition: false, scopeLine: 2, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!10 = !DISubroutineType(types: !11)
!11 = !{!3, !12, !3}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!13 = !{!0}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 1, !"Debug Info Version", i32 3}
!16 = !{!"clang version 3.5.0 "}
!17 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 7, type: !18, isLocal: false, isDefinition: true, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !4, retainedNodes: !5)
!18 = !DISubroutineType(types: !19)
!19 = !{!3}
!20 = !DILocation(line: 8, scope: !17)
!21 = !DILocalVariable(name: "this", arg: 1, scope: !22, type: !23, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = distinct !DISubprogram(name: "func", linkageName: "_ZN3foo4funcEi", scope: !7, file: !2, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !4, declaration: !9, retainedNodes: !5)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!24 = !DIExpression()
!25 = !DILocation(line: 0, scope: !22, inlinedAt: !20)
!26 = !DILocalVariable(name: "x", arg: 2, scope: !22, file: !2, line: 2, type: !3)
!27 = !DILocation(line: 2, scope: !22, inlinedAt: !20)

