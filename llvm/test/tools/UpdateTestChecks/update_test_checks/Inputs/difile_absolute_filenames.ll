; RUN: opt < %s -S | FileCheck %s

define void @f() !dbg !5 {
entry:
  ret void
}

define void @non_absolute_filename_non_empty_directory() !dbg !8 {
entry:
  ret void
}

define void @absolute_filename_non_empty_directory() !dbg !10 {
entry:
  ret void
}

define void @non_absolute_filename_empty_directory() !dbg !12 {
entry:
  ret void
}

define void @absolute_filename_empty_directory() !dbg !14 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1)
!1 = !DIFile(filename: "file", directory: "")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !6, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "non_absolute_filename_non_empty_directory", scope: !9, file: !9, type: !6, unit: !0)
!9 = !DIFile(filename: "file.c", directory: "dir")
!10 = distinct !DISubprogram(name: "absolute_filename_non_empty_directory", scope: !11, file: !11, type: !6, unit: !0)
!11 = !DIFile(filename: "/abs/path/file.c", directory: "dir")
!12 = distinct !DISubprogram(name: "non_absolute_filename_empty_directory", scope: !13, file: !13, type: !6, unit: !0)
!13 = !DIFile(filename: "file.c", directory: "")
!14 = distinct !DISubprogram(name: "absolute_filename_empty_directory", scope: !15, file: !15, type: !6, unit: !0)
!15 = !DIFile(filename: "/abs/path/file.c", directory: "")
