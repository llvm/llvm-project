; RUN: opt -S -passes=asan -asan-use-stack-safety=0 -asan-skip-promotable-allocas=0 %s -o - | FileCheck %s
; Generated from:
; int bar(int y) {
;   return y + 2;
; }

source_filename = "/tmp/t.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

; Function Attrs: noinline nounwind optnone sanitize_address ssp uwtable
define i32 @foo(i32 %i) #0 !dbg !8 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %i.addr, metadata !12, metadata !DIExpression()), !dbg !13

  ; CHECK: %asan_local_stack_base = alloca i64
  ; CHECK: %[[PHI:.*]] = phi ptr {{.*}} %MyAlloca
  ; CHECK: store ptr %[[PHI]], ptr %asan_local_stack_base
; CHECK: #dbg_declare(ptr %asan_local_stack_base, [[VAR_I:![0-9]+]], !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 32), [[LOC_I:![0-9]+]]
  %0 = load i32, ptr %i.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 2, !dbg !15
  ret i32 %add, !dbg !16
}

; CHECK: [[SCOPE:![0-9]+]] = distinct !DISubprogram(name: "foo"
; CHECK: [[VAR_I]] = !DILocalVariable(name: "i", arg: 1, scope: [[SCOPE]]
; CHECK: [[LOC_I]] = !DILocation(line: 1, column: 13, scope: [[SCOPE]]

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone sanitize_address ssp uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 320115) (llvm/trunk 320116)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/t.c", directory: "/Data/llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 320115) (llvm/trunk 320116)"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "i", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!13 = !DILocation(line: 1, column: 13, scope: !8)
!14 = !DILocation(line: 2, column: 10, scope: !8)
!15 = !DILocation(line: 2, column: 12, scope: !8)
!16 = !DILocation(line: 2, column: 3, scope: !8)
