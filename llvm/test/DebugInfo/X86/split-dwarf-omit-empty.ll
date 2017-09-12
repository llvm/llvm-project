; RUN: %llc_dwarf -split-dwarf-file=foo.dwo  %s -filetype=obj -o - | llvm-dwarfdump -debug-info - | FileCheck %s

; Created from:
; a.cpp:
;   void f1();
;   inline __attribute__((always_inline)) __attribute__((used)) void f2() { f1(); }
; b.cpp:
;   void f2();
;   void f3() {
;     f2();
;   }
; $ clang++ -fno-split-dwarf-inlining {a,b}.cpp -emit-llvm -S -g
; $ llvm-link {a,b}.ll -S -o ab.ll
; Then strip out the @llvm.used global, so no out of line definition of 'f2'
; will be emitted. This emulates something more like the available_externally
; import performed by ThinLTO.

; CHECK: Compile Unit
; CHECK-NOT: Compile Unit

target triple = "x86_64-pc-linux"

declare void @_Z2f1v()

; Function Attrs: noinline norecurse uwtable
define i32 @main() !dbg !9 {
entry:
  call void @_Z2f1v(), !dbg !13
  ret i32 0, !dbg !18
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 304054) (llvm/trunk 304080)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false)
!1 = !DIFile(filename: "a.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 304054) (llvm/trunk 304080)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false)
!4 = !DIFile(filename: "b.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!5 = !{!"clang version 5.0.0 (trunk 304054) (llvm/trunk 304080)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "main", scope: !4, file: !4, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !3, variables: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 2, column: 73, scope: !14, inlinedAt: !17)
!14 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !15, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = distinct !DILocation(line: 3, column: 3, scope: !9)
!18 = !DILocation(line: 4, column: 1, scope: !9)
