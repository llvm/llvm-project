; Check that there is no persistent state in the MachO emitter that crashes
; us when reusing the pass manager.
; RUN: llc -mtriple=x86_64-apple-darwin -compile-twice -filetype=obj %s -o -

; Force the creation of a DWARF section
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
