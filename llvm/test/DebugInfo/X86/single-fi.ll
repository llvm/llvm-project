; RUN: llc -mtriple=x86_64-apple-darwin -o - %s -filetype=obj \
; RUN:   | llvm-dwarfdump -v -debug-info - | FileCheck %s
; A single FI location. This used to trigger an assertion in debug libstdc++.
; CHECK: DW_TAG_formal_parameter
;                                          fbreg -8
; CHECK-NEXT: DW_AT_location {{.*}} (DW_OP_fbreg -8)
; CHECK-NEXT: DW_AT_name {{.*}} "dipsy"
define void @tinkywinky(ptr %dipsy) !dbg !6 {
entry:
  %dipsy.addr = alloca ptr
  store ptr %dipsy, ptr %dipsy.addr
  call void @llvm.dbg.declare(metadata ptr %dipsy.addr, metadata !12, metadata
!13), !dbg !14
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 297917) (llvm/trunk 297929)", isOptimized: false,
runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "teletubbies.c", directory: "/home/davide/work/llvm/build-clang/bin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0 (trunk 297917) (llvm/trunk 297929)"}
!6 = distinct !DISubprogram(name: "tinkywinky", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags:
DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DILocalVariable(name: "dipsy", arg: 1, scope: !6, file: !1, line: 1, type: !9)
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 29, scope: !6)
!15 = !DILocation(line: 1, column: 37, scope: !6)
