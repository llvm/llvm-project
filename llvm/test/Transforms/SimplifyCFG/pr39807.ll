; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -hoist-common-insts=true < %s | FileCheck %s

declare void @personality()

define void @test(i1 %b) personality void()* @personality !dbg !1 {
; CHECK:      invoke void @inlinable()
; CHECK-NEXT:    to label %common.ret unwind label %failure, !dbg ![[DBGLOC:[0-9]+]]
    br i1 %b, label %if, label %else

if:
    invoke void @inlinable()
        to label %success unwind label %failure, !dbg !2

else:
    invoke void @inlinable()
        to label %success unwind label %failure, !dbg !8

success:
    ret void

failure:
    landingpad {}
        cleanup
    ret void
}

define internal void @inlinable() !dbg !7 {
    ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5, !6}

; CHECK: ![[DBGLOC]] = !DILocation(line: 0
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, runtimeVersion: 0, file: !3)
!1 = distinct !DISubprogram(name: "test", unit: !0)
!2 = !DILocation(line: 2, scope: !1)
!3 = !DIFile(filename: "foo", directory: ".")
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "inlinable", unit: !0)
!8 = !DILocation(line: 3, scope: !1)
