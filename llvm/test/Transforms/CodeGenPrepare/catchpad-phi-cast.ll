; RUN: opt -codegenprepare -S < %s | FileCheck %s

; The following target lines are needed for the test to exercise what it should.
; Without these lines, CodeGenPrepare does not try to sink the bitcasts.
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @__CxxFrameHandler3(...)

declare void @f()

declare void @g(i8*)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

; CodeGenPrepare will want to sink these bitcasts, but it selects the catchpad
; blocks as the place to which the bitcast should be sunk.  Since catchpads
; do not allow non-phi instructions before the terminator, this isn't possible. 

; CHECK-LABEL: @test(
define void @test(i32* %addr) personality i32 (...)* @__CxxFrameHandler3 {
; CHECK: entry:
; CHECK-NEXT: %x = getelementptr i32, i32* %addr, i32 1
; CHECK-NEXT: %p1 = bitcast i32* %x to i8*
entry:
  %x = getelementptr i32, i32* %addr, i32 1
  %p1 = bitcast i32* %x to i8*
  invoke void @f()
          to label %invoke.cont unwind label %catch1

; CHECK: invoke.cont:
; CHECK-NEXT: %y = getelementptr i32, i32* %addr, i32 2
; CHECK-NEXT: %p2 = bitcast i32* %y to i8*
invoke.cont:
  %y = getelementptr i32, i32* %addr, i32 2
  %p2 = bitcast i32* %y to i8*
  invoke void @f()
          to label %done unwind label %catch2

done:
  ret void

catch1:
  %cp1 = catchpad [] to label %catch.dispatch unwind label %catchend1

catch2:
  %cp2 = catchpad [] to label %catch.dispatch unwind label %catchend2

; CHECK: catch.dispatch:
; CHECK-NEXT: %p = phi i8* [ %p1, %catch1 ], [ %p2, %catch2 ]
catch.dispatch:
  %p = phi i8* [ %p1, %catch1 ], [ %p2, %catch2 ]
  call void @g(i8* %p)
  unreachable

catchend1:
  catchendpad unwind to caller

catchend2:
  catchendpad unwind to caller
}

; CodeGenPrepare will want to hoist these llvm.dbg.value calls to the phi, but
; there is no insertion point in a catchpad block.

; CHECK-LABEL: @test_dbg_value(
define void @test_dbg_value() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %a = alloca i8
  %b = alloca i8
  invoke void @f() to label %next unwind label %catch.dispatch
next:
  invoke void @f() to label %ret unwind label %catch.dispatch
ret:
  ret void

catch.dispatch:
  %p = phi i8* [%a, %entry], [%b, %next]
  %cp1 = catchpad [] to label %catch unwind label %catchend

catch:
  tail call void @llvm.dbg.value(metadata i8* %p, i64 0, metadata !11, metadata !13), !dbg !14
  invoke void @g(i8* %p) to label %catchret unwind label %catchend
catchret:
  catchret %cp1 to label %ret

; CHECK: catch.dispatch:
; CHECK-NEXT: phi i8
; CHECK-NEXT: catchpad
; CHECK-NOT: llvm.dbg.value

; CHECK: catch:
; CHECK-NEXT: call void @llvm.dbg.value

catchend:
  catchendpad unwind to caller
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 254906) (llvm/trunk 254917)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: null, subprograms: !3)
!1 = !DIFile(filename: "t.c", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!3 = !{!4}
!4 = distinct !DISubprogram(name: "test_dbg_value", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, variables: null)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.8.0 (trunk 254906) (llvm/trunk 254917)"}
!11 = !DILocalVariable(name: "p", scope: !4, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!13 = !DIExpression()
!14 = !DILocation(line: 2, column: 8, scope: !4)
!15 = !DILocation(line: 3, column: 1, scope: !4)
