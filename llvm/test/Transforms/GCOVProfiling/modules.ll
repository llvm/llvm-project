; RUN: opt -insert-gcov-profiling -o - < %s | llvm-dis | FileCheck -check-prefix=EMIT-ARCS %s

; EMIT-ARCS-NOT: call void @llvm_gcda_start_file

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: false, runtimeVersion: 2, splitDebugFilename: "my.dwo", emissionKind: FullDebug, enums: !2, retainedTypes: !2, subprograms: !2, globals: !2, imports: !2, dwoId: 43981)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
