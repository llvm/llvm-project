; RUN: llc < %s -O0 -filetype=obj -o - -emit-heterogeneous-dwarf-as-user-ops=false | llvm-dwarfdump - | FileCheck --check-prefixes=CHECK,CHECK-ORIG-OPS %s
; RUN: llc < %s -O0 -filetype=obj -o - | llvm-dwarfdump - | FileCheck --check-prefixes=CHECK,CHECK-USER-OPS %s

; CHECK: DW_TAG_variable
; CHECK: DW_AT_name ("x")
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_addr 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_form_aspace_address)
; CHECK-USER-OPS: DW_AT_location (DW_OP_addr 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg.def !0

!llvm.dbg.cu = !{!1}
!llvm.dbg.retainedNodes = !{!3}
!llvm.module.flags = !{!7, !8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DIFragment()
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 6c601a5f5b6a3d32552bdc5eeb9f3ca6e516e8e6)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "-", directory: "/")
!3 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpArg(0, i32*), DIOpDeref(i32)), argObjects: {!0})
!4 = distinct !DIGlobalVariable(name: "x", scope: !1, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
!5 = !DIFile(filename: "<stdin>", directory: "/")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 4}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{!"clang"}
