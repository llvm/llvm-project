; RUN: llc -mtriple=aarch64-linux-gnu -enable-machine-outliner -filetype=obj %s -o %t
; RUN: llvm-objdump --disassemble --no-show-raw-insn %t > %t.dump
; RUN: llvm-dwarfdump --debug-line %t >> %t.dump
; RUN: FileCheck %s < %t.dump

; Capture each outlined call sequence and the instruction after it. Check that
; each sequence has one continuous source line and that the following
; instruction returns to its own line. The shared outlined function is
; artificial and legitimately has line 0.

; CHECK-LABEL: <caller>:
; CHECK:      [[#%x,FIRST_SAVE:]]:{{.*}}mov [[SAVE_REG:x[0-9]+]], x30
; CHECK-NEXT: {{[0-9a-f]+}}:{{.*}}bl{{.*}}<OUTLINED_FUNCTION_0>
; CHECK-NEXT: {{[0-9a-f]+}}:{{.*}}mov x30, [[SAVE_REG]]
; CHECK-NEXT: [[#%x,FIRST_AFTER:]]:{{.*}}add
; CHECK:      [[#%x,SECOND_SAVE:]]:{{.*}}mov [[SAVE_REG]], x30
; CHECK-NEXT: {{[0-9a-f]+}}:{{.*}}bl{{.*}}<OUTLINED_FUNCTION_0>
; CHECK-NEXT: {{[0-9a-f]+}}:{{.*}}mov x30, [[SAVE_REG]]
; CHECK-NEXT: [[#%x,SECOND_AFTER:]]:{{.*}}subs
; CHECK:      [[#%x,OUTLINED:]] <OUTLINED_FUNCTION_0>:

; CHECK: Address            Line
; CHECK:      0x[[#%.16x,FIRST_SAVE]]     10
; CHECK-NEXT: 0x[[#%.16x,FIRST_AFTER]]     11
; CHECK-NEXT: 0x[[#%.16x,SECOND_SAVE]]     20
; CHECK-NEXT: 0x[[#%.16x,SECOND_AFTER]]     21
; CHECK:      0x[[#%.16x,OUTLINED]]      0

define void @caller() #0 !dbg !6 {
  %slot1 = alloca i32, align 4
  %slot2 = alloca i32, align 4
  %slot3 = alloca i32, align 4
  %slot4 = alloca i32, align 4
  store i32 0, ptr %slot1, align 4, !dbg !7
  store i32 0, ptr %slot2, align 4, !dbg !7
  store i32 0, ptr %slot3, align 4, !dbg !7
  store i32 0, ptr %slot4, align 4, !dbg !7

  %v1 = load i32, ptr %slot1, align 4, !dbg !8
  %v2 = add nsw i32 %v1, 1, !dbg !8
  store i32 %v2, ptr %slot1, align 4, !dbg !8
  %v3 = load i32, ptr %slot3, align 4, !dbg !8
  %v4 = add nsw i32 %v3, 1, !dbg !8
  store i32 %v4, ptr %slot3, align 4, !dbg !8
  %v5 = load i32, ptr %slot4, align 4, !dbg !8
  %v6 = add nsw i32 %v5, 1, !dbg !8
  store i32 %v6, ptr %slot4, align 4, !dbg !8
  %v7 = load i32, ptr %slot2, align 4, !dbg !8
  %v8 = add nsw i32 %v7, 1, !dbg !9
  store i32 %v8, ptr %slot2, align 4, !dbg !9

  %v9 = load i32, ptr %slot1, align 4, !dbg !10
  %v10 = add nsw i32 %v9, 1, !dbg !10
  store i32 %v10, ptr %slot1, align 4, !dbg !10
  %v11 = load i32, ptr %slot3, align 4, !dbg !10
  %v12 = add nsw i32 %v11, 1, !dbg !10
  store i32 %v12, ptr %slot3, align 4, !dbg !10
  %v13 = load i32, ptr %slot4, align 4, !dbg !10
  %v14 = add nsw i32 %v13, 1, !dbg !10
  store i32 %v14, ptr %slot4, align 4, !dbg !10
  %v15 = load i32, ptr %slot2, align 4, !dbg !10
  %v16 = add nsw i32 %v15, -1, !dbg !11
  store i32 %v16, ptr %slot2, align 4, !dbg !11
  ret void, !dbg !12
}

attributes #0 = { noinline noredzone nounwind optnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !2)
!6 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILocation(line: 2, column: 1, scope: !6)
!8 = !DILocation(line: 10, column: 1, scope: !6)
!9 = !DILocation(line: 11, column: 1, scope: !6)
!10 = !DILocation(line: 20, column: 1, scope: !6)
!11 = !DILocation(line: 21, column: 1, scope: !6)
!12 = !DILocation(line: 22, column: 1, scope: !6)
