; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck --check-prefix=CHECK --check-prefix=CHECK-UNMAT %s
; RUN: llvm-as < %s | llvm-dis -materialize-metadata | FileCheck --check-prefix=CHECK-UNMAT %s
; RUN: verify-uselistorder %s

; CHECK-UNMAT: @global = global i32 0, !foo [[M2:![0-9]+]], !foo [[M3:![0-9]+]], !baz [[M3]]
@global = global i32 0, !foo !2, !foo !3, !baz !3

; CHECK-LABEL: @test
; CHECK: ret void, !foo [[M0:![0-9]+]], !bar [[M1:![0-9]+]]
define void @test() !dbg !1 {
  add i32 2, 1, !bar !0
  add i32 1, 2, !foo !1
  call void @llvm.dbg.func.start(metadata !"foo")
  extractvalue {{i32, i32}, i32} undef, 0, 1, !foo !0
  ret void, !foo !0, !bar !1
}

; CHECK: define void @test2() !foo [[M2]] !baz [[M3]]
define void @test2() !foo !2 !baz !3 {
  unreachable
}

; CHECK: define void @test3() !bar [[M3]]
; CHECK: unreachable, !bar [[M4:![0-9]+]]
define void @test3() !bar !3 {
  unreachable, !bar !4
}

; CHECK-LABEL: define void @test_attachment_name() {
; CHECK:   unreachable, !\342abc [[M4]]
define void @test_attachment_name() {
  ;; Escape the first character when printing text IR, since it's a digit
  unreachable, !\34\32abc !4
}

; CHECK-UNMAT: [[M2]] = distinct !{}
; CHECK-UNMAT: [[M3]] = distinct !{}
; CHECK: [[M0]] = !DILocation
; CHECK: [[M1]] = distinct !DISubprogram
; CHECK: [[M4]] = distinct !{}

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
