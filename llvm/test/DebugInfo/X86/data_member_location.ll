; RUN: llc -mtriple=x86_64-linux -O0 -o - -filetype=obj < %s | llvm-dwarfdump -debug-info -| FileCheck %s
; RUN: llc -mtriple=x86_64-linux -dwarf-version=2 -O0 -o - -filetype=obj < %s | llvm-dwarfdump -debug-info -| FileCheck -check-prefix=DWARF2 %s

; Generated from Clang with the following source:
;
; struct foo {
;   char c;
;   int i;
; };
; 
; foo f;

; CHECK: DW_AT_name {{.*}} "c"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_data_member_location {{.*}} (0x00)

; CHECK: DW_AT_name {{.*}} "i"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_data_member_location {{.*}} (0x04)

; DWARF2: DW_AT_name {{.*}} "c"
; DWARF2-NOT: DW_TAG
; DWARF2: DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)

; DWARF2: DW_AT_name {{.*}} "i"
; DWARF2-NOT: DW_TAG
; DWARF2: DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x4)

source_filename = "test/DebugInfo/X86/data_member_location.ll"

%struct.foo = type { i8, i32 }

@f = global %struct.foo zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 6, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "data_member_location.cpp", directory: "/tmp/dbginfo")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !2, line: 1, size: 64, align: 32, elements: !4, identifier: "_ZTS3foo")
!4 = !{!5, !7}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !3, file: !2, line: 2, baseType: !6, size: 8, align: 8)
!6 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !3, file: !2, line: 3, baseType: !8, size: 32, align: 32, offset: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, retainedTypes: !11, globals: !12, imports: !10)
!10 = !{}
!11 = !{!3}
!12 = !{!0}
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.4 "}

