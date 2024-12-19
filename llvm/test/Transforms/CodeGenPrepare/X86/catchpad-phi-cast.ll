; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S < %s | FileCheck %s
; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S < %s --try-experimental-debuginfo-iterators | FileCheck %s

; The following target lines are needed for the test to exercise what it should.
; Without these lines, CodeGenPrepare does not try to sink the bitcasts.
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @__CxxFrameHandler3(...)

declare void @f()

declare void @g(ptr)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

; CodeGenPrepare will want to sink these bitcasts, but it selects the catchpad
; blocks as the place to which the bitcast should be sunk.  Since catchpads
; do not allow non-phi instructions before the terminator, this isn't possible.

; CHECK-LABEL: @test(
define void @test(ptr %addr) personality ptr @__CxxFrameHandler3 {
entry:
  %x = getelementptr i32, ptr %addr, i32 1
  invoke void @f()
          to label %invoke.cont unwind label %catch1

; CHECK: invoke.cont:
; CHECK-NEXT: %y = getelementptr i32, ptr %addr, i32 2
invoke.cont:
  %y = getelementptr i32, ptr %addr, i32 2
  invoke void @f()
          to label %done unwind label %catch2

done:
  ret void

catch1:
  %cs1 = catchswitch within none [label %handler1] unwind to caller

handler1:
  %cp1 = catchpad within %cs1 []
  br label %catch.shared
; CHECK: handler1:
; CHECK-NEXT: catchpad within %cs1

catch2:
  %cs2 = catchswitch within none [label %handler2] unwind to caller

handler2:
  %cp2 = catchpad within %cs2 []
  br label %catch.shared
; CHECK: handler2:
; CHECK: catchpad within %cs2

; CHECK: catch.shared:
; CHECK-NEXT: %p = phi ptr [ %x, %handler1 ], [ %y, %handler2 ]
catch.shared:
  %p = phi ptr [ %x, %handler1 ], [ %y, %handler2 ]
  call void @g(ptr %p)
  unreachable
}

; CodeGenPrepare will want to hoist these llvm.dbg.value calls to the phi, but
; there is no insertion point in a catchpad block.

; CHECK-LABEL: @test_dbg_value(
define void @test_dbg_value() personality ptr @__CxxFrameHandler3 {
entry:
  %a = alloca i8
  %b = alloca i8
  invoke void @f() to label %next unwind label %catch.dispatch
next:
  invoke void @f() to label %ret unwind label %catch.dispatch
ret:
  ret void

catch.dispatch:
  %p = phi ptr [%a, %entry], [%b, %next]
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  %cp1 = catchpad within %cs1 []
  tail call void @llvm.dbg.value(metadata ptr %p, i64 0, metadata !11, metadata !13), !dbg !14
  call void @g(ptr %p)
  catchret from %cp1 to label %ret

; CHECK: catch.dispatch:
; CHECK-NEXT: phi ptr
; CHECK-NEXT: catchswitch
; CHECK-NOT: #dbg_value

; CHECK: catch:
; CHECK-NEXT: catchpad
; CHECK-NEXT: #dbg_value
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 254906) (llvm/trunk 254917)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: null)
!1 = !DIFile(filename: "t.c", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = distinct !DISubprogram(name: "test_dbg_value", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: null)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.8.0 (trunk 254906) (llvm/trunk 254917)"}
!11 = !DILocalVariable(name: "p", scope: !4, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!13 = !DIExpression(DW_OP_deref)
!14 = !DILocation(line: 2, column: 8, scope: !4)
!15 = !DILocation(line: 3, column: 1, scope: !4)
