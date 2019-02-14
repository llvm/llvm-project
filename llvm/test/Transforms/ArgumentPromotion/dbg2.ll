; RUN: opt < %s -argpromotion -instcombine -S | FileCheck %s

%f_ty = type void (i8*)*

define void @foo() {
entry:
  %f_p = getelementptr inbounds %f_ty, %f_ty* null, i32 0
  store %f_ty @bar, %f_ty* %f_p, align 1
  ret void
}

define internal void @bar(i8*) !dbg !1 {
entry:
  ret void
}

; The new copy should get the !dbg metadata
; CHECK: define internal void @bar() !dbg
; The old copy should now be a declaration without any !dbg metadata
; CHECK-NOT: declare dso_local void @0(i8*) !dbg
; CHECK: declare dso_local void @0(i8*)

!llvm.dbg.cu = !{}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(name: "bar", scope: !2, file: !2, line: 14, type: !3, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !5)
!2 = !DIFile(filename: "foo.c", directory: "/bar")
!3 = !DISubroutineType(types: !4)
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "My Compiler", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
