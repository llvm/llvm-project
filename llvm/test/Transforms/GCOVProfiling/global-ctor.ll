;; For a global constructor, _GLOBAL__sub_I_ only has artificial lines.
;; Test that we don't instrument those functions.
; RUN: mkdir -p %t && cd %t
; RUN: opt -S -passes=insert-gcov-profiling < %s | FileCheck %s

@var = dso_local global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_a.cc, ptr null }]

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !7 {
; CHECK: define internal void @__cxx_global_var_init()
; CHECK: @__llvm_gcov_ctr
; CHECK: call i32 @_Z3foov()
entry:
  %call = call i32 @_Z3foov(), !dbg !9
  store i32 %call, ptr @var, align 4, !dbg !9
  ret void, !dbg !9
}

declare i32 @_Z3foov()

;; Artificial lines only. Don't instrument.
define internal void @_GLOBAL__sub_I_a.cc() section ".text.startup" !dbg !10 {
; CHECK: define internal void @_GLOBAL__sub_I_a.cc()
; CHECK-NOT: @__llvm_gcov_ctr
; CHECK: call void @__cxx_global_var_init()
entry:
  call void @__cxx_global_var_init(), !dbg !11
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "a.cc", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 11, scope: !7)
!10 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_a.cc", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 0, scope: !10)
