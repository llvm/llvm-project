; RUN: llc -filetype=asm -asm-verbose=0 < %s | FileCheck %s

; int main()
; {
;     int x = 0;
;     if (x > 0)
;         return x;
;     x = -1; // <== this line should have correct debug location
;     return -1;
; }

; CHECK: .loc 1 6 7
; CHECK: mvn

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

; Function Attrs: nounwind
define i32 @main() !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata i32* %x, metadata !10, metadata !11), !dbg !12
  store i32 0, i32* %x, align 4, !dbg !12
  %0 = load i32, i32* %x, align 4, !dbg !13
  %cmp = icmp sgt i32 %0, 0, !dbg !15
  br i1 %cmp, label %if.then, label %if.end, !dbg !16

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %x, align 4, !dbg !17
  store i32 %1, i32* %retval, !dbg !18
  br label %return, !dbg !18

if.end:                                           ; preds = %entry
  store i32 -1, i32* %x, align 4, !dbg !19
  store i32 -1, i32* %retval, !dbg !20
  br label %return, !dbg !20

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, i32* %retval, !dbg !21
  ret i32 %2, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/home/user/clang/build")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 3, type: !7)
!11 = !DIExpression()
!12 = !DILocation(line: 3, column: 9, scope: !4)
!13 = !DILocation(line: 4, column: 9, scope: !14)
!14 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 9)
!15 = !DILocation(line: 4, column: 11, scope: !14)
!16 = !DILocation(line: 4, column: 9, scope: !4)
!17 = !DILocation(line: 5, column: 13, scope: !14)
!18 = !DILocation(line: 5, column: 9, scope: !14)
!19 = !DILocation(line: 6, column: 7, scope: !4)
!20 = !DILocation(line: 7, column: 5, scope: !4)
!21 = !DILocation(line: 8, column: 1, scope: !4)
