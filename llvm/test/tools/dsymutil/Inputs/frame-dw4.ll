; Generated from frame.c on Darwin with '-arch i386 -gdwarf-4 -emit-llvm'
; ModuleID = 'frame.c'
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.11.0"

; Function Attrs: nounwind ssp
define i32 @bar(i32 %b) #0 !dbg !4 {
entry:
  %b.addr = alloca i32, align 4
  %var = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !13, metadata !14), !dbg !15
  call void @llvm.dbg.declare(metadata i32* %var, metadata !16, metadata !14), !dbg !17
  %0 = load i32, i32* %b.addr, align 4, !dbg !18
  %add = add nsw i32 %0, 1, !dbg !19
  store i32 %add, i32* %var, align 4, !dbg !17
  %call = call i32 @foo(i32* %var), !dbg !20
  ret i32 %call, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @foo(i32*) #2

; Function Attrs: nounwind ssp
define i32 @baz(i32 %b) #0 !dbg !8 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !22, metadata !14), !dbg !23
  %0 = load i32, i32* %b.addr, align 4, !dbg !24
  %call = call i32 @bar(i32 %0), !dbg !25
  ret i32 %call, !dbg !26
}

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="yonah" "target-features"="+cx16,+sse,+sse2,+sse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="yonah" "target-features"="+cx16,+sse,+sse2,+sse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 239176) (llvm/trunk 239190)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "frame.c", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = !{!"clang version 3.7.0 (trunk 239176) (llvm/trunk 239190)"}
!13 = !DILocalVariable(name: "b", arg: 1, scope: !4, file: !1, line: 3, type: !7)
!14 = !DIExpression()
!15 = !DILocation(line: 3, column: 13, scope: !4)
!16 = !DILocalVariable(name: "var", scope: !4, file: !1, line: 4, type: !7)
!17 = !DILocation(line: 4, column: 6, scope: !4)
!18 = !DILocation(line: 4, column: 12, scope: !4)
!19 = !DILocation(line: 4, column: 14, scope: !4)
!20 = !DILocation(line: 5, column: 9, scope: !4)
!21 = !DILocation(line: 5, column: 2, scope: !4)
!22 = !DILocalVariable(name: "b", arg: 1, scope: !8, file: !1, line: 8, type: !7)
!23 = !DILocation(line: 8, column: 13, scope: !8)
!24 = !DILocation(line: 9, column: 13, scope: !8)
!25 = !DILocation(line: 9, column: 9, scope: !8)
!26 = !DILocation(line: 9, column: 2, scope: !8)
