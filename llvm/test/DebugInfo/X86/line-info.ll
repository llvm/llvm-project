; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; CHECK: [[FILEID:[0-9]+]]]{{.*}}list0.h
; CHECK: [[FILEID]]      0      1   0  0 0 is_stmt{{$}}

; IR generated from clang -g -emit-llvm with the following source:
; list0.h:
; int foo (int x) {
;     return ++x;
; }
; list0.c:
; #include "list0.h"
; int main() {
; }

define i32 @foo(i32 %x) #0 !dbg !4 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !14, metadata !DIExpression()), !dbg !15
  %0 = load i32, ptr %x.addr, align 4, !dbg !16
  %inc = add nsw i32 %0, 1, !dbg !16
  store i32 %inc, ptr %x.addr, align 4, !dbg !16
  ret i32 %inc, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define i32 @main() #0 !dbg !10 {
entry:
  ret i32 0, !dbg !17
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports:  !2)
!1 = !DIFile(filename: "list0.c", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !5, scope: !6, type: !7, retainedNodes: !2)
!5 = !DIFile(filename: "./list0.h", directory: "/usr/local/google/home/blaikie/dev/scratch")
!6 = !DIFile(filename: "./list0.h", directory: "/usr/local/google/home/blaikie/dev/scratch")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "main", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !11, type: !12, retainedNodes: !2)
!11 = !DIFile(filename: "list0.c", directory: "/usr/local/google/home/blaikie/dev/scratch")
!12 = !DISubroutineType(types: !13)
!13 = !{!9}
!14 = !DILocalVariable(name: "x", line: 1, arg: 1, scope: !4, file: !6, type: !9)
!15 = !DILocation(line: 1, scope: !4)
!16 = !DILocation(line: 2, scope: !4)
!17 = !DILocation(line: 3, scope: !18)
!18 = !DILexicalBlockFile(discriminator: 0, file: !11, scope: !10)
!19 = !{i32 1, !"Debug Info Version", i32 3}
