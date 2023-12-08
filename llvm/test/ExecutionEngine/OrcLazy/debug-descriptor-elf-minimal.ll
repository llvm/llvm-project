; REQUIRES: native && target-x86_64

; RUN: lli --jit-linker=rtdyld \
; RUN:     --generate=__dump_jit_debug_descriptor %s | FileCheck %s
;
; RUN: lli --jit-linker=jitlink \
; RUN:     --generate=__dump_jit_debug_descriptor %s | FileCheck %s
;
; CHECK: Reading __jit_debug_descriptor at 0x{{.*}}
; CHECK: Version: 1
; CHECK: Action: JIT_REGISTER_FN
; CHECK:       Entry               Symbol File             Size  Previous Entry
; CHECK: [ 0]  0x{{.*}}            0x{{.*}}              {{.*}}  0x0000000000000000

target triple = "x86_64-unknown-unknown-elf"

; Built-in symbol provided by the JIT
declare void @__dump_jit_debug_descriptor(ptr)

; Host-process symbol from the GDB JIT interface
@__jit_debug_descriptor = external global i8, align 1

define i32 @main() !dbg !9 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @__dump_jit_debug_descriptor(ptr @__jit_debug_descriptor), !dbg !13
  ret i32 0, !dbg !14
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!llvm.ident = !{!8}

!0 = !{i32 2, !"SDK Version", [3 x i32] [i32 10, i32 15, i32 6]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "compiler version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, nameTableKind: None)
!6 = !DIFile(filename: "source-file.c", directory: "/workspace")
!7 = !{}
!8 = !{!"compiler version"}
!9 = distinct !DISubprogram(name: "main", scope: !6, file: !6, line: 4, type: !10, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 5, column: 3, scope: !9)
!14 = !DILocation(line: 6, column: 3, scope: !9)
