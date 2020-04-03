; RUN: opt -debugify -S < %s | FileCheck --check-prefixes=ALL,VALUE %s
; RUN: opt -debugify -debugify-level=locations -S < %s | FileCheck --check-prefixes=ALL --implicit-check-not=dbg.value %s

; ALL-LABEL: @test
define void @test() {
  %add = add i32 1, 2
; ALL-NEXT:  %add = add i32 1, 2, !dbg [[L1:![0-9]+]]
; VALUE-NEXT: call void @llvm.dbg.value(metadata i32 %add, metadata [[add:![0-9]+]], metadata !DIExpression()), !dbg [[L1]]
  %sub = sub i32 %add, 1
; ALL-NEXT: %sub = sub i32 %add, 1, !dbg [[L2:![0-9]+]]
; VALUE-NEXT: call void @llvm.dbg.value(metadata i32 %sub, metadata [[sub:![0-9]+]], metadata !DIExpression()), !dbg [[L2]]
; ALL-NEXT: ret void, !dbg [[L3:![0-9]+]]
  ret void
}

; VALUE: [[add]] = !DILocalVariable
; VALUE: [[sub]] = !DILocalVariable

; ALL: [[L1]] = !DILocation(line: 1, column: 1, scope:
; ALL: [[L2]] = !DILocation(line: 2, column: 1, scope:
; ALL: [[L3]] = !DILocation(line: 3, column: 1, scope:
