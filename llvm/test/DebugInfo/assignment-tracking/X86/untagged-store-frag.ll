; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_
; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Hand-written to test untagged store handling on a simple case. Here's what
;; we're looking at in the IR:

;; 1. mem(a): bits [0,  64) = !14
;; 2. dbg(a): bits [0,  64) = !14 ; Use memory loc
;; 3. dbg(a): bits [0,  32) = <unique ID> ; Use implicit loc, dbg.value has no ID
;; 4. dbg(a): bits [32, 64) = !16 ; These bits don't use mem loc.
;;                                ; Linked to a def that comes later ---+
;; ...                                                                ; |
;; 5. mem(a): bits [0,  32) = <unique ID> ; Untagged store            ; |
;; ..                                                                 ; |
;; 6. mem(a): bits [32, 64) = !16 ; <-----------------------------------+

;; Taking the '<number>.' above as the 'position', check we get defs that look
;; like this:
;; Position | bits [0, 32) | bits [32,  64)
;; ---------+--------------+---------------
;; 2.       | Mem          | Mem
;; 3.       | Value        | Mem
;; 4.       | Value        | Value
;; 5.       | Mem          | Value
;; 6.       | Mem          | Mem

; CHECK-DAG: ![[A:[0-9]+]] = !DILocalVariable(name: "a",

; CHECK:      DBG_VALUE %stack.0.a.addr, $noreg, ![[A]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: ADJCALLSTACKDOWN
; CHECK-NEXT: @step
; CHECK-NEXT: ADJCALLSTACKUP
; CHECK-NEXT: DBG_VALUE 5,               $noreg, ![[A]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE $noreg,          $noreg, ![[A]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT:   MOV32mi %stack.0.a.addr, 1, $noreg, 0, $noreg, 123
; CHECK-NEXT: DBG_VALUE %stack.0.a.addr, $noreg, ![[A]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT:   MOV32mr %stack.0.a.addr, 1, $noreg, 4, $noreg, %1 :: (store (s32) into %ir.add.ptr, align 8)
; CHECK-NEXT: DBG_VALUE %stack.0.a.addr, $noreg, ![[A]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)

target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef i64 @_Z1fl(i64 noundef %a, i32 %b) #0 !dbg !8 {
entry:
  %a.addr = alloca i64, align 8, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(), metadata !13, metadata ptr %a.addr, metadata !DIExpression()), !dbg !15
  call void @step()
  call void @llvm.dbg.value(metadata i64 5, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !15
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !16, metadata ptr %a.addr, metadata !DIExpression()), !dbg !15
  %frag.addr = bitcast ptr %a.addr to ptr
  store i32 123, ptr %frag.addr, align 8
  %0 = bitcast ptr %a.addr to ptr
  %add.ptr = getelementptr inbounds i32, ptr %0, i64 1
  store i32 %b, ptr %add.ptr, align 8, !DIAssignID !16
  %1 = load i64, ptr %a.addr, align 8
  ret i64 %1
}

declare void @step()
declare void @llvm.dbg.value(metadata, metadata, metadata) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !1000}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "f", linkageName: "_Z1fl", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !{}
!13 = distinct !DIAssignID()
!14 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocation(line: 0, scope: !8)
!16 = distinct !DIAssignID()
!17 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
