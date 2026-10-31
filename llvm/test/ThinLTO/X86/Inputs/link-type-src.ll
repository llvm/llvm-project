target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%T2 = type { double, double, i32, i32 }

define void @src_f(ptr byval(%T2) %p) {
  ret void, !dbg !3
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "src.cpp", directory: "")
!3 = !DILocation(line: 1, scope: !4)
!4 = distinct !DISubprogram(name: "g", scope: !2, file: !2, type: !5, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Shared", identifier: "_ZTS6Shared")
