; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; Test the DW_AT_producer DWARG attribute.
; When producer and flags are both given in DIComileUnit, set DW_AT_producer
; as two values combined.
;
; The test splits into two parts, this is LLVM part. The frontend part can be
; found at llvm/tools/clang/test/Driver/debug-options.c.
;
; Generated and reduced from:
; clang++ -g -grecord-gcc-switches test.cc -S -llvm-emit -o -
;
; test.cc:
;   int main() {
;     return 0;
;   }

; CHECK: DW_AT_producer
; CHECK-SAME: "clang++ -g -grecord-gcc-switches test.cc -S -emit-llvm -o -"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() !dbg !6 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang++", isOptimized: false, flags: "-g -grecord-gcc-switches test.cc -S -emit-llvm -o -", runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cc", directory: "d")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang"}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 5, column: 3, scope: !6)
