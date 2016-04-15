; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK-LABEL: @test
; CHECK: ret void, !bar !4, !foo !3
define void @test() !dbg !1 {
  add i32 2, 1, !bar !0
  add i32 1, 2, !foo !1
  call void @llvm.dbg.func.start(metadata !"foo")
  extractvalue {{i32, i32}, i32} undef, 0, 1, !foo !0
  ret void, !foo !0, !bar !1
}

; CHECK-LABEL: define void @test2() !foo !5 !baz !6
define void @test2() !foo !2 !baz !3 {
  unreachable
}

; CHECK-LABEL: define void @test3() !bar !6
; CHECK: unreachable, !bar !7
define void @test3() !bar !3 {
  unreachable, !bar !4
}

; CHECK-LABEL: define void @test_attachment_name() {
; CHECK:   unreachable, !\342abc !7
define void @test_attachment_name() {
  ;; Escape the first character when printing text IR, since it's a digit
  unreachable, !\34\32abc !4
}

!llvm.module.flags = !{!7}
!llvm.dbg.cu = !{!5}
!0 = !DILocation(line: 662302, column: 26, scope: !1)
!1 = distinct !DISubprogram(name: "foo", isDefinition: true, unit: !5)
!2 = distinct !{}
!3 = distinct !{}
!4 = distinct !{}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !6,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!6 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!7 = !{i32 2, !"Debug Info Version", i32 3}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

!foo = !{ !0 }
!bar = !{ !1 }
