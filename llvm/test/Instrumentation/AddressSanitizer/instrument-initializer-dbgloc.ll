; RUN: opt < %s -passes=asan -S | FileCheck %s

@glob = internal global i32 0, align 4, sanitize_address_dyninit

define internal void @__late_ctor() sanitize_address section ".text.startup" !dbg !5 {
entry:
  ret void, !dbg !10
}

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__late_ctor, ptr null }]

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "__late_ctor", scope: !1, file: !1, line: 1, type: !6, isLocal: true, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !{}
!10 = !DILocation(line: 2, column: 1, scope: !5)

; CHECK-LABEL: define internal void @__late_ctor
; CHECK: call void @__asan_before_dynamic_init(i64 ptrtoint (ptr @___asan_gen_module to i64)), !dbg ![[DBG1:[0-9]+]]
; CHECK: call void @__asan_after_dynamic_init(), !dbg ![[DBG2:[0-9]+]]
; CHECK: ret void, !dbg ![[DBG2]]
; CHECK: ![[DBG1]] = !DILocation(line: 1, scope: ![[SP:[0-9]+]])
; CHECK: ![[DBG2]] = !DILocation(line: 2, column: 1, scope: ![[SP]])
