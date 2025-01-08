; RUN: opt < %s -passes=jump-table-to-switch -pass-remarks=jump-table-to-switch -S -o /dev/null 2>&1 | FileCheck %s

; CHECK: remark: /tmp/tmp.cc:2:20: expanded indirect call into switch

@func_array = constant [2 x ptr] [ptr @func0, ptr @func1]

define i32 @func0() {
  ret i32 1
}

define i32 @func1() {
  ret i32 2
}

define i32 @function_with_jump_table(i32 %index) {
  %gep = getelementptr inbounds [2 x ptr], ptr @func_array, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep
  %result = call i32 %func_ptr(), !dbg !8
  ret i32 %result
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 18.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/tmp.cc", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 18.0.0 "}
!6 = distinct !DISubprogram(name: "success", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 20, scope: !6)
!9 = !DILocation(line: 2, column: 21, scope: !6)
!10 = !DILocation(line: 2, column: 22, scope: !6)
