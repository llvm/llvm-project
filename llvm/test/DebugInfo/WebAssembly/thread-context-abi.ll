; Ensure that using libcall thread context with an empty function produces a frame base 
; that uses a local, and that using the global thread context produces a frame base that 
; uses the __stack_pointer global.

; Test generated via: clang --target=wasm32-unknown-unknown-wasm foo.c -g -O2
; void foo() {}

; RUN: llc < %s -filetype=obj -mtriple=wasm32-wasip3 -o - | llvm-dwarfdump - | FileCheck %s --check-prefix=LIBCALL
; RUN: llc < %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s --check-prefix=GLOBAL

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define hidden void @foo() local_unnamed_addr #0 !dbg !9 {
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+bulk-memory,+bulk-memory-opt,+call-indirect-overlong,+multivalue,+mutable-globals,+nontrapping-fptoint,+reference-types,+sign-ext" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang version 23.0.0"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!10 = !DISubroutineType(types: !11)
!11 = !{null}

; LIBCALL: DW_AT_frame_base        (DW_OP_WASM_location 0x0 0x0, DW_OP_stack_value)
; GLOBAL: DW_AT_frame_base        (DW_OP_WASM_location 0x3 0x0, DW_OP_stack_value)
