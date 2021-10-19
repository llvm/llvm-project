; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   extern int do_work(int) __attribute__((section(".callback_fn")));
;   long bpf_helper(void *callback_fn);
;   long prog() {
;       return bpf_helper(&do_work);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

; Function Attrs: nounwind
define dso_local i64 @prog() local_unnamed_addr #0 !dbg !7 {
entry:
  %call = tail call i64 @bpf_helper(i8* bitcast (i32 (i32)* @do_work to i8*)) #2, !dbg !11
  ret i64 %call, !dbg !12
}

; CHECK:             .long   0                               # BTF_KIND_FUNC_PROTO(id = 4)
; CHECK-NEXT:        .long   218103809                       # 0xd000001
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   51                              # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   55                              # BTF_KIND_FUNC(id = 6)
; CHECK-NEXT:        .long   201326594                       # 0xc000002
; CHECK-NEXT:        .long   4

; CHECK:             .long   74                              # BTF_KIND_DATASEC(id = 10)
; CHECK-NEXT:        .long   251658241                       # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   do_work
; CHECK-NEXT:        .long   0

; CHECK:             .ascii  "int"                           # string offset=51
; CHECK:             .ascii  "do_work"                       # string offset=55
; CHECK:             .ascii  ".callback_fn"                  # string offset=74

declare !dbg !13 dso_local i64 @bpf_helper(i8*) local_unnamed_addr #1

declare !dbg !17 dso_local i32 @do_work(i32) #1 section ".callback_fn"

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git aa382ed8a38d5efa118e1b2617544f5c253658a9)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/btf/core")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git aa382ed8a38d5efa118e1b2617544f5c253658a9)"}
!7 = distinct !DISubprogram(name: "prog", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!11 = !DILocation(line: 4, column: 12, scope: !7)
!12 = !DILocation(line: 4, column: 5, scope: !7)
!13 = !DISubprogram(name: "bpf_helper", scope: !1, file: !1, line: 2, type: !14, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!10, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!17 = !DISubprogram(name: "do_work", scope: !1, file: !1, line: 1, type: !18, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
