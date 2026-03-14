; RUN: opt < %s -passes=asan -asan-use-after-return=never -asan-use-stack-safety=0 -S | FileCheck %s

; Checks that llvm.dbg.declare instructions are updated
; accordingly as we merge allocas.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G = global [32 x i8] zeroinitializer, align 32, !dbg !0, !type !6
; CHECK: @G = global { [32 x i8], [32 x i8] } zeroinitializer{{(, comdat)?}}, align 32, !dbg !0, !type [[TYPE:![0-9]+]]

define i32 @_Z3zzzi(i32 %p) nounwind uwtable sanitize_address !dbg !12 {
entry:
  %p.addr = alloca i32, align 4
  %r = alloca i32, align 4
  store volatile i32 %p, ptr %p.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %p.addr, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %r, metadata !19, metadata !DIExpression()), !dbg !21
  %0 = load i32, ptr %p.addr, align 4, !dbg !21
  %add = add nsw i32 %0, 1, !dbg !21
  store volatile i32 %add, ptr %r, align 4, !dbg !21
  %1 = load i32, ptr %r, align 4, !dbg !22
  ret i32 %1, !dbg !22
}

;   CHECK: define i32 @_Z3zzzi
;   CHECK: [[MyAlloca:%.*]] = alloca i8, i64 64
; Note: these dbg.declares used to contain `ptrtoint` operands. The instruction
; selector would then decline to put the variable in the MachineFunction side
; table. Check that the dbg.declares have `alloca` operands.
;   CHECK: #dbg_declare(ptr [[MyAlloca]], ![[ARG_ID:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 32),
;   CHECK: #dbg_declare(ptr [[MyAlloca]], ![[VAR_ID:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 48),

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "G", type: !2, isLocal: false, isDefinition: true)
!2 = !DICompositeType(tag: DW_TAG_array_type, baseType: !3, size: 256, elements: !4)
!3 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!4 = !{!5}
!5 = !DISubrange(count: 32)
!6 = !{i64 0, !"G"}
; CHECK: [[TYPE]] = !{i64 0, !"G"}

!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 169314)", isOptimized: true, emissionKind: FullDebug, file: !23, enums: !8, retainedTypes: !8, globals: !8)
!8 = !{}
!12 = distinct !DISubprogram(name: "zzz", linkageName: "_Z3zzzi", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !7, scopeLine: 1, file: !23, scope: !13, type: !14, retainedNodes: !8)
!13 = !DIFile(filename: "a.cc", directory: "/usr/local/google/llvm_cmake_clang/tmp/debuginfo")
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !16}
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DILocalVariable(name: "p", line: 1, arg: 1, scope: !12, file: !13, type: !16)
!18 = !DILocation(line: 1, scope: !12)
!19 = !DILocalVariable(name: "r", line: 2, scope: !20, file: !13, type: !16)

; Verify that debug descriptors for argument and local variable will be replaced
; with descriptors that end with OpDeref (encoded as 2).
;   CHECK: ![[ARG_ID]] = !DILocalVariable(name: "p", arg: 1,{{.*}} line: 1
;   CHECK: ![[VAR_ID]] = !DILocalVariable(name: "r",{{.*}} line: 2
; Verify that there are no more variable descriptors.
;   CHECK-NOT: !DILocalVariable(tag: DW_TAG_arg_variable
;   CHECK-NOT: !DILocalVariable(tag: DW_TAG_auto_variable


!20 = distinct !DILexicalBlock(line: 1, column: 0, file: !23, scope: !12)
!21 = !DILocation(line: 2, scope: !20)
!22 = !DILocation(line: 3, scope: !20)
!23 = !DIFile(filename: "a.cc", directory: "/usr/local/google/llvm_cmake_clang/tmp/debuginfo")
!24 = !{i32 1, !"Debug Info Version", i32 3}
