; RUN: llc -filetype=asm --aarch64-enable-sink-fold=true %s -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "aarch64--linux-gnueabihf"

; int
; main(void)
; {
;     return -1;
; }

; CHECK: test.c:4:5
; CHECK: mov

; Function Attrs: nounwind
define i32 @main() !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  ret i32 -1, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/home/user/clang/build")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !DILocation(line: 4, column: 5, scope: !4)
