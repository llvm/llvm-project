; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o %t.o
; RUN: llvm-dwarfdump --verify %t.o
; RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o %t.o
; RUN: llvm-dwarfdump --verify %t.o
; RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s

; Both setup paths join before the first source statement on line 3.
; A function breakpoint must stop there, after value has been stored in its
; stack home and setup_ready has been initialized.
;
; CHECK:      Address Line Column
; CHECK-NOT:  prologue_end
; CHECK:      0x{{[0-9a-f]+}} 3 5 {{.*}} is_stmt prologue_end
; CHECK-NOT:  prologue_end
; CHECK:      end_sequence

@setup_ready = global i64 0
@choose = global i8 0

define i64 @probe(i64 %value) noinline !dbg !5 {
entry:
  %home = alloca i64, align 8
  store i64 %value, ptr %home, align 8
  #dbg_declare(ptr %home, !9, !DIExpression(), !10)
  %flag = load volatile i8, ptr @choose
  %condition = icmp ne i8 %flag, 0
  br i1 %condition, label %parent, label %root
parent:
  %parent_value = load volatile i64, ptr @setup_ready
  br label %body
root:
  br label %body
body:
  %seed = phi i64 [ %parent_value, %parent ], [ 1, %root ]
  store volatile i64 %seed, ptr @setup_ready
  %incoming = load i64, ptr %home, align 8, !dbg !12
  %local = add i64 %incoming, 19, !dbg !12
  %twice = mul i64 %local, 2, !dbg !13
  ret i64 %twice, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "entry.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "probe", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !{!8, !8})
!8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "value", arg: 1, scope: !5, file: !1, line: 2, type: !8)
!10 = !DILocation(line: 0, scope: !5)
!12 = !DILocation(line: 3, column: 5, scope: !5)
!13 = !DILocation(line: 4, column: 5, scope: !5)
!15 = !DILocation(line: 5, column: 5, scope: !5)
