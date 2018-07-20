; RUN: llc -mtriple=x86_64-pc-linux-gnu -debugger-tune=gdb -filetype=asm < %s -no-dwarf-pub-sections | FileCheck %s --check-prefix=DISABLED
; RUN: llc -mtriple=x86_64-pc-linux-gnu -debugger-tune=gdb -filetype=asm < %s | FileCheck %s

; DISABLED-NOT: {{.debug_pubnames|.debug_pubtypes}}
; CHECK-DAG: .section .debug_pubnames
; CHECK-DAG: .section .debug_pubtypes

%struct.bar = type { %"struct.ns::foo" }
%"struct.ns::foo" = type { i8 }

@b = global %struct.bar zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, gnuPubnames: false)
!3 = !DIFile(filename: "type.cpp", directory: "/tmp/dbginfo")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !3, line: 5, size: 8, elements: !7, identifier: "_ZTS3bar")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !6, file: !3, line: 6, baseType: !9, size: 8)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", scope: !10, file: !3, line: 2, size: 8, elements: !4, identifier: "_ZTSN2ns3fooE")
!10 = !DINamespace(name: "ns", scope: null)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang"}

