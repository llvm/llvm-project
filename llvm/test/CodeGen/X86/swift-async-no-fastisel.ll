; RUN: llc %s --fast-isel=true --stop-after=finalize-isel -o %t \
; RUN:   -experimental-debug-variable-locations=false --global-isel=false
; RUN:   FileCheck %s < %t
; RUN:   FileCheck %s --check-prefix=INTRINSICS < %t


source_filename = "ir_x86.ll"
target triple = "x86_64-*"

define swifttailcc void @foo(ptr swiftasync %0) !dbg !43 {
  call void asm sideeffect "", "r"(ptr %0), !dbg !62
  ; FastISEL doesn't preserve %0 here. Check that this function is lowered with SelectionDAG.
  call void @llvm.dbg.value(metadata ptr %0, metadata !54, metadata !DIExpression(DW_OP_plus_uconst, 4242)), !dbg !62
  ret void, !dbg !62
}

; CHECK-NOT: DBG_VALUE $noreg
; INTRINSICS: ![[VAR:[0-9]*]] = !DILocalVariable(name: "msg",
; INTRINSICS: DBG_VALUE {{.*}}, ![[VAR]], !DIExpression(DW_OP_plus_uconst, 4242)


declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.dbg.cu = !{!16}

!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 2}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, producer: "blah", emissionKind: FullDebug)
!17 = !DIFile(filename: "blah", directory: "blah")
!43 = distinct !DISubprogram(name: "blah", linkageName: "blah", file: !17, line: 87, type: !44, scopeLine: 87, unit: !16, retainedNodes: !48)
!44 = !DISubroutineType(types: !45)
!45 = !{!46}
!46 = !DICompositeType(tag: DW_TAG_structure_type, name: "blah")
!48 = !{!54}
!54 = !DILocalVariable(name: "msg", arg: 1, scope: !43, file: !17, line: 87, type: !46)
!62 = !DILocation(line: 87, column: 30, scope: !43)
