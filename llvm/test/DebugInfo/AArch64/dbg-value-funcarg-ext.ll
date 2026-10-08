; RUN: llc -mtriple=aarch64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=true | FileCheck %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=false | FileCheck %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -filetype=obj -o - %s | llvm-dwarfdump --name=lazy - | FileCheck %s --check-prefix=DWARF

; A C bool parameter is stored as i8, so its dbg.value refers to a zext of the
; argument rather than to the argument. With AAPCS64 the parameter is a plain
; i1, but the caller zero-extends a bool to 8 bits, which argument lowering
; records with ISD::AssertZextBool. The low 8 bits of the argument register
; then hold the value of the zext from the function entry on: describe the
; parameter by the argument register so that it has a location at the entry.
;
;   void ext(void);
;   void zext_bool(char *p, _Bool lazy) { ext(); *p = lazy; }

; CHECK: ![[LAZY:[0-9]+]] = !DILocalVariable(name: "lazy", arg: 2
; CHECK-LABEL: name: zext_bool
; CHECK: DBG_VALUE $w1, $noreg, ![[LAZY]], !DIExpression()

; DWARF:      DW_AT_location
; DWARF-NEXT: [0x0000000000000000, {{.*}}): DW_OP_reg1 W1
; DWARF:      DW_AT_name ("lazy")

declare void @ext()

define void @zext_bool(ptr %p, i1 %lazy) !dbg !7 {
entry:
  %frombool = zext i1 %lazy to i8
    #dbg_value(i8 %frombool, !13, !DIExpression(), !15)
  call void @ext(), !dbg !16
  store i8 %frombool, ptr %p, align 1, !dbg !16
  ret void, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "")
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !5}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)

!7 = distinct !DISubprogram(name: "zext_bool", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!12 = !{!13}
!13 = !DILocalVariable(name: "lazy", arg: 2, scope: !7, file: !1, line: 2, type: !5)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 2, column: 40, scope: !7)
