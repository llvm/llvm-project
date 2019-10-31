; RUN: llc -stop-before=finalize-isel -o - %s -mtriple=i386-- | FileCheck %s --check-prefix=MIR
; RUN: llc -o - %s -mtriple=i386-- --filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF --implicit-check-not=DW_TAG_subprogram
; REQUIRES: object-emission
;
; Test that, when arguments are passed on the stack (such as i386),
; variable location dereferences occur in the right place. When referring to
; argument stack slots a deref must be used to load the slot first.

; MIR: ![[FOOVAR:[0-9]+]] = !DILocalVariable(name: "foovar"
; MIR: ![[BARVAR:[0-9]+]] = !DILocalVariable(name: "barvar"
; MIR: ![[BAZVAR:[0-9]+]] = !DILocalVariable(name: "bazvar"

; Plain i32 on the stack.
; MIR-LABEL: name: foo
; MIR:       DBG_VALUE %fixed-stack.0, $noreg, ![[FOOVAR]],
; MIR-SAME:  !DIExpression(DW_OP_deref)
; DWARF: DW_TAG_subprogram
; DWARF-LABEL: DW_AT_name ("cheese")
; DWARF:       DW_TAG_variable
; DWARF-NEXT:  DW_AT_location (DW_OP_fbreg +4)
; DWARF-NEXT:  DW_AT_name ("foovar")
define i8 @foo(i32 %blah) !dbg !20 {
entry:
  call void @llvm.dbg.value(metadata i32 %blah, metadata !23, metadata !DIExpression()), !dbg !21
  ret i8 0, !dbg !21
}

; Pointer on the stack that we fiddle with.
; MIR-LABEL: name: bar
; MIR:       DBG_VALUE %fixed-stack.0, $noreg, ![[BARVAR]],
; MIR-SAME:  !DIExpression(DW_OP_deref_size, 4, DW_OP_plus_uconst, 4, DW_OP_stack_value)
; DWARF: DW_TAG_subprogram
; DWARF-LABEL: DW_AT_name ("nope")
; DWARF:       DW_TAG_variable
; DWARF-NEXT:  DW_AT_location (DW_OP_fbreg +4, DW_OP_deref_size 0x4, DW_OP_plus_uconst 0x4, DW_OP_stack_value)
; DWARF-NEXT:  DW_AT_name ("barvar")
define i8 @bar(i32 *%blah) !dbg !30 {
entry:
  call void @llvm.dbg.value(metadata i32* %blah, metadata !33, metadata !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)), !dbg !31
  ret i8 0, !dbg !31
}

; Pointer that we use as a dbg.declare variable location, after fiddling with
; the pointer value.
; MIR-LABEL: name: baz
; MIR:       DBG_VALUE %fixed-stack.0, $noreg, ![[BAZVAR]],
; MIR-SAME:  !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4, DW_OP_deref)
; DWARF: DW_TAG_subprogram
; DWARF-LABEL: DW_AT_name ("brains")
; DWARF:       DW_TAG_variable
; DWARF-NEXT:  DW_AT_location (DW_OP_fbreg +4, DW_OP_deref, DW_OP_plus_uconst 0x4)
; DWARF-NEXT:  DW_AT_name ("bazvar")
define i8 @baz(i32 *%blah) !dbg !40 {
entry:
  call void @llvm.dbg.declare(metadata i32* %blah, metadata !43, metadata !DIExpression(DW_OP_plus_uconst, 4)), !dbg !41
  ret i8 0, !dbg !41
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "asdf", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "nil", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !DISubroutineType(types: !2)
!8 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)

!20 = distinct !DISubprogram(name: "cheese", linkageName: "cheese", scope: null, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !0, retainedNodes: !22)
!21 = !DILocation(line: 1, column: 1, scope: !20)
!22 = !{!23}
!23 = !DILocalVariable(name: "foovar", scope: !20, file: !1, line: 14, type: !8)

!30 = distinct !DISubprogram(name: "nope", linkageName: "nope", scope: null, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !0, retainedNodes: !32)
!31 = !DILocation(line: 1, column: 1, scope: !30)
!32 = !{!33}
!33 = !DILocalVariable(name: "barvar", scope: !30, file: !1, line: 14, type: !8)

!40 = distinct !DISubprogram(name: "brains", linkageName: "brains", scope: null, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !0, retainedNodes: !42)
!41 = !DILocation(line: 1, column: 1, scope: !40)
!42 = !{!43}
!43 = !DILocalVariable(name: "bazvar", scope: !40, file: !1, line: 14, type: !8)
