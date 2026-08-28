; RUN: llc -mtriple=riscv32 -mattr=+save-restore -verify-machineinstrs < %s | FileCheck %s
;
; Document the current shrink-wrapping behavior with and without a debug
; instruction in the common return block.

declare ptr @llvm.stacksave.p0()
declare void @llvm.stackrestore.p0(ptr)
declare void @notdead(ptr)

define void @without_debug(i32 %n) nounwind {
; CHECK-LABEL: without_debug:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    li a1, 32
; CHECK-NEXT:    bltu a1, a0, .LBB0_2
; CHECK-NEXT:  # %bb.1: # %if.then
; CHECK-NEXT:    call t0, __riscv_save_{{[0-9]+}}
; CHECK:         tail __riscv_restore_{{[0-9]+}}
; CHECK:       .LBB0_2: # %if.end
; CHECK-NEXT:    ret
entry:
  %cmp = icmp ult i32 %n, 33
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %size = zext i32 %n to i64
  %stack = call ptr @llvm.stacksave.p0()
  %buffer = alloca i8, i64 %size, align 16
  call void @notdead(ptr %buffer)
  call void @llvm.stackrestore.p0(ptr %stack)
  br label %if.end

if.end:
  ret void
}

define void @with_debug(i32 %n) nounwind !dbg !4 {
; CHECK-LABEL: with_debug:
; CHECK:       # %bb.0: # %entry
; CHECK:         call t0, __riscv_save_{{[0-9]+}}
; CHECK:         li a1, 32
; CHECK:         bltu a1, a0, .LBB1_2
; CHECK:       # %bb.1: # %if.then
; CHECK-NOT:     __riscv_save
; CHECK:       .LBB1_2: # %if.end
; CHECK:         tail __riscv_restore_{{[0-9]+}}
entry:
  %cmp = icmp ult i32 %n, 33
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %size = zext i32 %n to i64
  %stack = call ptr @llvm.stacksave.p0()
  %buffer = alloca i8, i64 %size, align 16
  call void @notdead(ptr %buffer)
  call void @llvm.stackrestore.p0(ptr %stack)
  br label %if.end

if.end:
    #dbg_value(i32 0, !8, !DIExpression(), !9)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "with_debug", scope: !1, file: !1, line: 1, type: !5, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !11)
!6 = !{!8}
!8 = !DILocalVariable(name: "ghost", scope: !4, file: !1, line: 2, type: !2)
!9 = !DILocation(line: 0, scope: !4)
!10 = !DILocation(line: 3, scope: !4)
!11 = !{null, !2}
