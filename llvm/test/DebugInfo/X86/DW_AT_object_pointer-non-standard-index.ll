; Similar to DW_AT_object_pointer.ll but tests that we correctly
; encode the object pointer index even if it's not the first argument
; of the subprogram (which isn't something the major compilers do,
; but is not mandated by DWARF).

; RUN: llc -mtriple=x86_64-apple-darwin -debugger-tune=lldb -dwarf-version=5 -filetype=obj < %s | \
; RUN:      llvm-dwarfdump -v -debug-info - | FileCheck %s --check-prefixes=CHECK

; CHECK: DW_TAG_class_type
; CHECK: [[DECL:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:                         DW_AT_name {{.*}} "A"
; CHECK: DW_AT_object_pointer [DW_FORM_implicit_const] (2)
;
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_object_pointer [DW_FORM_ref4]     (cu + 0x{{[0-9a-f]*}} => {[[PARAM:0x[0-9a-f]*]]})
; CHECK:   DW_AT_specification [DW_FORM_ref4] (cu + {{.*}} => {[[DECL]]}
; CHECK:   DW_TAG_formal_parameter
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: "this"
; CHECK: [[PARAM]]: DW_TAG_formal_parameter
; CHECK: DW_AT_name
; CHECK-SAME: = "this")
; CHECK:   DW_TAG_formal_parameter

%class.A = type { i8 }

define linkonce_odr noundef ptr @_ZN1AC1Eii(ptr noundef nonnull returned align 1 dereferenceable(1) %this, i32 noundef %x, i32 noundef %y, i32 noundef %z) !dbg !24 {
entry:
  %this.addr = alloca ptr, align 8
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %z.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !26, !DIExpression(), !28)
  store i32 %x, ptr %x.addr, align 4
    #dbg_declare(ptr %x.addr, !29, !DIExpression(), !30)
  store i32 %y, ptr %y.addr, align 4
    #dbg_declare(ptr %y.addr, !31, !DIExpression(), !32)
  store i32 %z, ptr %y.addr, align 4
    #dbg_declare(ptr %z.addr, !36, !DIExpression(), !37)
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = load i32, ptr %x.addr, align 4, !dbg !33
  %1 = load i32, ptr %y.addr, align 4, !dbg !33
  %2 = load i32, ptr %z.addr, align 4, !dbg !33
  ret ptr %this1, !dbg !34
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 20.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "object_ptr.cpp", directory: "/tmp")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !6, identifier: "_ZTS1A")
!6 = !{!7}
!7 = !DISubprogram(name: "A", scope: !5, file: !3, line: 2, type: !8, scopeLine: 2, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !11, !11, !10, !35}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 20.0.0git"}
!24 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC1Eii", scope: !5, file: !3, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !7, retainedNodes: !25)
!25 = !{}
!26 = !DILocalVariable(name: "this", arg: 3, scope: !24, type: !27, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!28 = !DILocation(line: 0, scope: !24)
!29 = !DILocalVariable(name: "x", arg: 2, scope: !24, file: !3, line: 2, type: !11)
!30 = !DILocation(line: 2, column: 19, scope: !24)
!31 = !DILocalVariable(name: "y", arg: 1, scope: !24, file: !3, line: 2, type: !11)
!32 = !DILocation(line: 2, column: 26, scope: !24)
!33 = !DILocation(line: 2, column: 29, scope: !24)
!34 = !DILocation(line: 2, column: 30, scope: !24)
!35 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!36 = !DILocalVariable(name: "z", arg: 4, scope: !24, file: !3, line: 2, type: !35)
!37 = !DILocation(line: 2, column: 35, scope: !24)
