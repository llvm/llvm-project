; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   extern int foo(void);
;   static __attribute__((noinline)) int test1() { return foo(); }
;   int test2() { return test1(); }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

; Function Attrs: nounwind
define dso_local i32 @test2() local_unnamed_addr #0 !dbg !12 {
entry:
  %call = tail call fastcc i32 @test1(), !dbg !13
  ret i32 %call, !dbg !14
}
; Function Attrs: noinline nounwind
define internal fastcc i32 @test1() unnamed_addr #1 !dbg !15 {
entry:
  %call = tail call i32 @foo() #3, !dbg !16
  ret i32 %call, !dbg !17
}
declare !dbg !4 dso_local i32 @foo() local_unnamed_addr #2

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   88
; CHECK-NEXT:        .long   88
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   5                       # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 4)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   54                      # BTF_KIND_FUNC(id = 5)
; CHECK-NEXT:        .long   201326592               # 0xc000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 6)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   60                      # BTF_KIND_FUNC(id = 7)
; CHECK-NEXT:        .long   201326594               # 0xc000002
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "test2"                 # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=11
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/bugs/test.c" # string offset=17
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "test1"                 # string offset=54
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "foo"                   # string offset=60
; CHECK-NEXT:        .byte   0

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 000f95c4157aee07bd4ffc3f59ffdb6c7ecae4af)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/bugs")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 000f95c4157aee07bd4ffc3f59ffdb6c7ecae4af)"}
!12 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 3, column: 22, scope: !12)
!14 = !DILocation(line: 3, column: 15, scope: !12)
!15 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 2, column: 55, scope: !15)
!17 = !DILocation(line: 2, column: 48, scope: !15)
