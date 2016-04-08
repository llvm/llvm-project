; RUN: llc < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo() #0 !dbg !4 {
entry:
  call void @llvm.eh.unwind.init(), !dbg !9
  ret void, !dbg !10
}

; CHECK: @foo
; CHECK-NOT: .cfi_offset vrsave
; CHECK: blr

; Function Attrs: nounwind
declare void @llvm.eh.unwind.init() #0

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "/tmp/unwind-dw2.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "/tmp/unwind-dw2.c", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 3}
!9 = !DILocation(line: 2, scope: !4)
!10 = !DILocation(line: 3, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 3}
