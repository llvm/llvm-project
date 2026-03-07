; RUN: llc -O0 -mtriple=aarch64-linux-gnu -filetype=obj < %s \
; RUN:     | llvm-dwarfdump - | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK:   DW_AT_name      ("tls_var")
; CHECK-NEXT:   DW_AT_type      (0x{{.*}} "int")
; CHECK-NEXT:   DW_AT_external  (true)
; CHECK-NEXT:   DW_AT_decl_file ("{{.*}}tls-location.c")
; CHECK-NEXT:   DW_AT_decl_line (1)
; CHECK-NEXT:   DW_AT_location  (DW_OP_const8u 0x0, DW_OP_GNU_push_tls_address)

@tls_var = hidden thread_local global i32 0, align 4, !dbg !0

define i32 @foo() !dbg !10 {
entry:
  %0 = load i32, ptr @tls_var, align 4, !dbg !13
  ret i32 %0, !dbg !14
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = !DIGlobalVariableExpression(var: !2, expr: !DIExpression())
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!2 = distinct !DIGlobalVariable(name: "tls_var", scope: !1, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!3 = !DIFile(filename: "tls-location.c", directory: "/tmp")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 3, type: !11, unit: !1)
!11 = !DISubroutineType(types: !12)
!12 = !{!5}
!13 = !DILocation(line: 4, column: 10, scope: !10)
!14 = !DILocation(line: 4, column: 3, scope: !10)
