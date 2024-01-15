define i32 @sub1(i32 %0) {
  %2 = add i32 %0, -1
  ret i32 %2
}

define i32 @main(i32 %0) {
  %2 = call i32 @sub1(i32 %0)
  ret i32 %2
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang 18.0.0git", emissionKind: FullDebug)
!2 = !DIFile(filename: "argc_sub1.c", directory: ".")
