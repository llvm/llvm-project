; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple -debugger-tune=lldb -asm-verbose -O1 -o - < %t.ll | FileCheck %s
; RUN: llc -mtriple=%triple -debugger-tune=gdb -asm-verbose -O1 -o - < %t.ll | FileCheck %s --check-prefix=DISABLE
; RUN: llc -mtriple=%triple -disable-fp-elim -debugger-tune=lldb -asm-verbose -O1 -o - < %t.ll | FileCheck %s --check-prefix=DISABLE

; CHECK: DW_AT_APPLE_omit_frame_ptr
; DISABLE-NOT: DW_AT_APPLE_omit_frame_ptr


define i32 @foo() nounwind ssp !dbg !1 {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 42, i32* %0, align 4, !dbg !0
  %1 = load i32, i32* %0, align 4, !dbg !0             ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !0
  br label %return, !dbg !0

return:                                           ; preds = %entry
  %retval1 = load i32, i32* %retval, !dbg !0           ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !7
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!12}
!9 = !{!1}

!0 = !DILocation(line: 2, scope: !1)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 2, file: !10, scope: null, type: !4)
!2 = !DIFile(filename: "a.c", directory: "/tmp")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: FullDebug, file: !10, enums: !11, retainedTypes: !11, imports:  null)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DILocation(line: 2, scope: !8)
!8 = distinct !DILexicalBlock(line: 2, column: 0, file: !10, scope: !1)
!10 = !DIFile(filename: "a.c", directory: "/tmp")
!11 = !{}
!12 = !{i32 1, !"Debug Info Version", i32 3}
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
