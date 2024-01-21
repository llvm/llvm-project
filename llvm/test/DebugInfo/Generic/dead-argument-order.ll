; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: %llc_dwarf --try-experimental-debuginfo-iterators -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s

; Built from the following source with clang -O1
; struct S { int i; };
; int function(struct S s, int i) { return s.i + i; }

; Due to the X86_64 ABI, 's' is passed in registers and once optimized, the
; entirety of 's' is never reconstituted, since only the int is required, and
; thus the variable's location is unknown/dead to debug info.

; Future/current work should enable us to describe partial variables, which, in
; this case, happens to be the entire variable.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name ("function")
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name ("s")
; CHECK-NOT: DW_TAG
; FIXME: Even though 's' is never reconstituted into a struct, the one member
; variable is still live and used, and so we should be able to describe 's's
; location as the location of that int.
; CHECK-NOT: DW_AT_location
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name ("i")


%struct.S = type { i32 }

; Function Attrs: nounwind readnone uwtable
define i32 @_Z8function1Si(i32 %s.coerce, i32 %i) #0 !dbg !9 {
entry:
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !14, metadata !DIExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 %i, metadata !15, metadata !DIExpression()), !dbg !20
  %add = add nsw i32 %i, %s.coerce, !dbg !20
  ret i32 %add, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "dead-argument-order.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", line: 1, size: 32, align: 32, file: !1, elements: !5, identifier: "_ZTS1S")
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "i", line: 1, size: 32, align: 32, file: !1, scope: !4, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "function", linkageName: "_Z8function1Si", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 2, file: !1, scope: !10, type: !11, retainedNodes: !13)
!10 = !DIFile(filename: "dead-argument-order.cpp", directory: "/tmp/dbginfo")
!11 = !DISubroutineType(types: !12)
!12 = !{!7, !4, !7}
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "s", line: 2, arg: 1, scope: !9, file: !10, type: !4)
!15 = !DILocalVariable(name: "i", line: 2, arg: 2, scope: !9, file: !10, type: !7)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.5.0 "}
!19 = !{ptr undef}
!20 = !DILocation(line: 2, scope: !9)

