; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source code:
;   unsigned long long load_byte(void *skb,
;       unsigned long long off) asm("llvm.bpf.load.byte");
;   unsigned long long test(void *skb) {
;     return load_byte(skb, 10);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

; Function Attrs: nounwind readonly
define dso_local i64 @test(i8* readonly %skb) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i8* %skb, metadata !17, metadata !DIExpression()), !dbg !18
  %call = tail call i64 @llvm.bpf.load.byte(i8* %skb, i64 10), !dbg !19
  ret i64 %call, !dbg !20
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   60
; CHECK-NEXT:        .long   60
; CHECK-NEXT:        .long   78
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 2)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   5                       # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   64                      # 0x40
; CHECK-NEXT:        .long   28                      # BTF_KIND_FUNC(id = 4)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "skb"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "long long unsigned int" # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "test"                  # string offset=28
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=33
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/extern/test.c" # string offset=39
; CHECK-NEXT:        .byte   0

; Function Attrs: nounwind readonly
declare !dbg !4 i64 @llvm.bpf.load.byte(i8*, i64) #1
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 907019d835895443b198afcd992c42c9d3478fdf)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/extern")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "load_byte", linkageName: "llvm.bpf.load.byte", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8, !7}
!7 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 907019d835895443b198afcd992c42c9d3478fdf)"}
!13 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !14, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!7, !8}
!16 = !{!17}
!17 = !DILocalVariable(name: "skb", arg: 1, scope: !13, file: !1, line: 3, type: !8)
!18 = !DILocation(line: 0, scope: !13)
!19 = !DILocation(line: 4, column: 10, scope: !13)
!20 = !DILocation(line: 4, column: 3, scope: !13)
