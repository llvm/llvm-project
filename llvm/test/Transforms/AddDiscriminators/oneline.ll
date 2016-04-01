; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Discriminator support for code that is written in one line:
; #1 int foo(int i) {
; #2   if (i == 3 || i == 5) return 100; else return 99;
; #3 }

; i == 3:     discriminator 0
; i == 5:     discriminator 1
; return 100: discriminator 2
; return 99:  discriminator 3

define i32 @_Z3fooi(i32 %i) #0 !dbg !4 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 %i, i32* %2, align 4, !tbaa !13
  call void @llvm.dbg.declare(metadata i32* %2, metadata !9, metadata !17), !dbg !18
  %3 = load i32, i32* %2, align 4, !dbg !19, !tbaa !13
  %4 = icmp eq i32 %3, 3, !dbg !21
  br i1 %4, label %8, label %5, !dbg !22

; <label>:5                                       ; preds = %0
  %6 = load i32, i32* %2, align 4, !dbg !23, !tbaa !13
; CHECK:  %6 = load i32, i32* %2, align 4, !dbg ![[THEN1:[0-9]+]],{{.*}}

  %7 = icmp eq i32 %6, 5, !dbg !24
; CHECK:  %7 = icmp eq i32 %6, 5, !dbg ![[THEN2:[0-9]+]]

  br i1 %7, label %8, label %9, !dbg !25
; CHECK:  br i1 %7, label %8, label %9, !dbg ![[THEN3:[0-9]+]]

; <label>:8                                       ; preds = %5, %0
  store i32 100, i32* %1, align 4, !dbg !26
; CHECK: store i32 100, i32* %1, align 4, !dbg ![[ELSE:[0-9]+]]

  br label %10, !dbg !26
; CHECK: br label %10, !dbg ![[ELSE]]

; <label>:9                                       ; preds = %5
  store i32 99, i32* %1, align 4, !dbg !27
; CHECK: store i32 99, i32* %1, align 4, !dbg ![[COMBINE:[0-9]+]]

  br label %10, !dbg !27
; CHECK: br label %10, !dbg ![[COMBINE]]

; <label>:10                                      ; preds = %9, %8
  %11 = load i32, i32* %1, align 4, !dbg !28
  ret i32 %11, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

; We should be able to add discriminators even in the absence of llvm.dbg.cu.
; When using sample profiles, the front end will generate line tables but it
; does not generate llvm.dbg.cu to prevent codegen from emitting debug info
; to the final binary.
; !llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 250915)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "a.cc", directory: "/usr/local/google/home/dehao/discr")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.8.0 (trunk 250915)"}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
!17 = !DIExpression()
!18 = !DILocation(line: 1, column: 13, scope: !4)
!19 = !DILocation(line: 2, column: 7, scope: !20)
!20 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 7)
!21 = !DILocation(line: 2, column: 9, scope: !20)
!22 = !DILocation(line: 2, column: 14, scope: !20)
!23 = !DILocation(line: 2, column: 17, scope: !20)
!24 = !DILocation(line: 2, column: 19, scope: !20)
!25 = !DILocation(line: 2, column: 7, scope: !4)
!26 = !DILocation(line: 2, column: 25, scope: !20)
!27 = !DILocation(line: 2, column: 42, scope: !20)
!28 = !DILocation(line: 3, column: 1, scope: !4)

; CHECK: ![[THEN1]] = !DILocation(line: 2, column: 17, scope: ![[THENBLOCK:[0-9]+]])
; CHECK: ![[THENBLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 1)
; CHECK: ![[THEN2]] = !DILocation(line: 2, column: 19, scope: ![[THENBLOCK]])
; CHECK: ![[THEN3]] = !DILocation(line: 2, column: 7, scope: ![[THENBLOCK]])
; CHECK: ![[ELSE]] = !DILocation(line: 2, column: 25, scope: ![[ELSEBLOCK:[0-9]+]])
; CHECK: ![[ELSEBLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 2)
; CHECK: ![[COMBINE]] = !DILocation(line: 2, column: 42, scope: ![[COMBINEBLOCK:[0-9]+]])
; CHECK: ![[COMBINEBLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 3)
