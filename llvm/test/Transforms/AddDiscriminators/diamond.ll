; RUN: opt < %s -add-discriminators -S | FileCheck %s
; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; Discriminator support for diamond-shaped CFG.:
; #1 void bar(int);
; #2 
; #3 void foo(int i) {
; #4   if (i > 10)
; #5     bar(5); else bar(3);
; #6 }

; bar(5):     discriminator 0
; bar(3):     discriminator 1

; Function Attrs: uwtable
define void @_Z3fooi(i32 %i) #0 !dbg !4 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !11, metadata !12), !dbg !13
  %2 = load i32, i32* %1, align 4, !dbg !14
  %3 = icmp sgt i32 %2, 10, !dbg !16
  br i1 %3, label %4, label %5, !dbg !17

; <label>:4                                       ; preds = %0
  call void @_Z3bari(i32 5), !dbg !18
  br label %6, !dbg !18

; <label>:5                                       ; preds = %0
  call void @_Z3bari(i32 3), !dbg !19
; CHECK:  call void @_Z3bari(i32 3), !dbg ![[ELSE:[0-9]+]]
  br label %6

; <label>:6                                       ; preds = %5, %4
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z3bari(i32) #2

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 253273)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cc", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 253273)"}
!11 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 3, type: !7)
!12 = !DIExpression()
!13 = !DILocation(line: 3, column: 14, scope: !4)
!14 = !DILocation(line: 4, column: 7, scope: !15)
!15 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 7)
!16 = !DILocation(line: 4, column: 9, scope: !15)
!17 = !DILocation(line: 4, column: 7, scope: !4)
!18 = !DILocation(line: 5, column: 5, scope: !15)
!19 = !DILocation(line: 5, column: 18, scope: !15)
!20 = !DILocation(line: 6, column: 1, scope: !4)

; CHECK: ![[ELSE]] = !DILocation(line: 5, column: 18, scope: ![[ELSEBLOCK:[0-9]+]])
; CHECK: ![[ELSEBLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 1)
