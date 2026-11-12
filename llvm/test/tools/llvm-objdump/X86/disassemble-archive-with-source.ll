; This test checks that 'llvm-objdump --source' works with archive files.
; The reason this test is written in .ll is that showing source code
; requires the debug data to have the source file path. Since we create
; the source file ad-hoc in this test, we don't know the path beforehand to
; create the object and archive.

; RUN: rm -rf %t && mkdir -p %t/subdir && cd %t
; RUN: echo -e "int foo(int a)\n\n{ return a+1; }" > subdir/a.c
; RUN: sed -e "s,DIRNAME,%/t/subdir," %s | llc --filetype=obj -mtriple=x86_64-pc-linux -o a.o
; RUN: llvm-ar rc a.a a.o
; RUN: rm a.o
; RUN: llvm-objdump --source a.a | FileCheck %s

; CHECK: { return a+1; }

define i32 @foo(i32 %a) #0 !dbg !8 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !12, metadata !DIExpression()), !dbg !13
  %0 = load i32, ptr %a.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone sspstrong uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "DIRNAME")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 9.0.0"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!13 = !DILocation(line: 1, scope: !8)
!14 = !DILocation(line: 3, scope: !8)
