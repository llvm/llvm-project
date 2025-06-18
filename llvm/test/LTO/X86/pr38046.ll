; RUN: opt -module-summary -o %t.o %s
; RUN: llvm-lto2 run -save-temps -o %t.lto.o %t.o \
; RUN:   -r=%t.o,foo,plx \
; RUN:   -r=%t.o,get,pl
; RUN: llvm-dis %t.lto.o.0.2.internalize.bc >/dev/null 2>%t.dis.stderr || true
; RUN: FileCheck -allow-empty %s < %t.dis.stderr

; CHECK-NOT: Global is external, but doesn't have external or weak linkage

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
  call void @llvm.dbg.value(metadata ptr @get, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !6)
  ret i32 0
}

define i32 @get() {
  ret i32 0
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "t.cc", directory: "/tmp/t")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"ThinLTO", i32 0}
!6 = distinct !DISubprogram(unit: !0)
!7 = !DILocalVariable(name: "get", scope: !6)
