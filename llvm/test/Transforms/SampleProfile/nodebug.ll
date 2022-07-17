; RUN: opt %s -passes=sample-profile -sample-profile-file=%S/Inputs/nodebug.prof

define void @foo() !dbg !3 {
  call void @bar(), !dbg !4
  ret void
}

define void @bar() {
  call void @bar()
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "t", directory: "/tmp/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "a", scope: !1, file: !1, line: 10, unit: !0)
!4 = !DILocation(line: 10, scope: !3)
