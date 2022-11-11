; RUN: opt -S -passes=slp-vectorizer -mtriple=aarch64--linux-gnu -mcpu=generic -pass-remarks=slp-vectorizer -o /dev/null < %s 2>&1 | FileCheck %s

define void @f(double* %r, double* %w) {
  %r0 = getelementptr inbounds double, double* %r, i64 0
  %r1 = getelementptr inbounds double, double* %r, i64 1
  %f0 = load double, double* %r0
  %f1 = load double, double* %r1
  %add0 = fadd double %f0, %f0
  %add1 = fadd double %f1, %f1
  %w0 = getelementptr inbounds double, double* %w, i64 0
  %w1 = getelementptr inbounds double, double* %w, i64 1
; CHECK: remark: /tmp/s.c:5:10: Stores SLP vectorized with cost -4 and with tree size 3
  store double %add0, double* %w0, !dbg !9
  store double %add1, double* %w1
  ret void
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 281293) (llvm/trunk 281290)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 281293) (llvm/trunk 281290)"}
!7 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 5, column: 10, scope: !7)
