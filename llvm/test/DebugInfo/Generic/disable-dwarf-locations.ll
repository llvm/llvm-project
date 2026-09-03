; Check that -disable-dwarf-locations suppresses variable location information
; while leaving the rest of the debug info intact.

; RUN: llc -o - %s -filetype=obj \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -o - %s -filetype=obj -disable-dwarf-locations \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s --check-prefix=DISABLED

; DEFAULT:      DW_TAG_formal_parameter
; DEFAULT-NEXT:   DW_AT_location
; DEFAULT-NEXT:   DW_AT_name ("foo")

; The parameter, its name and its type are still described; only the location
; is gone.
; DISABLED:      DW_TAG_formal_parameter
; DISABLED-NEXT:   DW_AT_name ("foo")
; DISABLED-NOT:  DW_AT_location

define void @f(ptr %bar) !dbg !6 {
entry:
  %foo.addr = alloca ptr
  store ptr %bar, ptr %foo.addr
  call void @llvm.dbg.declare(metadata ptr %foo.addr, metadata !12, metadata !13), !dbg !14
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DILocalVariable(name: "foo", arg: 1, scope: !6, file: !1, line: 1, type: !10)
!13 = !DIExpression(DW_OP_deref)
!14 = !DILocation(line: 1, scope: !6)
!15 = !DILocation(line: 1, scope: !6)
