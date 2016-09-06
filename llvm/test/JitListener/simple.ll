; Verify the behavior of the IntelJITEventListener.
; RUN: llvm-jitlistener %s | FileCheck %s

; This test was created using the following file:
;
; 1: int foo(int a) {
; 2:   return a;
; 3: }
;

; CHECK: Method load [1]: foo, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: simple.c, line 1
; CHECK:   Line info @ {{[0-9]+}}: simple.c, line 2
; CHECK: Method unload [1]

; ModuleID = 'simple.c'

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !12, metadata !13), !dbg !14
  %0 = load i32, i32* %a.addr, align 4, !dbg !15
  ret i32 %0, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 (trunk)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "simple.c", directory: "F:\5Cusers\5Cakaylor\5Cllvm-s\5Cllvm\5Ctest\5CJitListener")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "simple.c", directory: "F:CusersCakaylorCllvm-sCllvmCtestCJitListener")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.6.0 (trunk)"}
!12 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 13, scope: !4)
!15 = !DILocation(line: 2, column: 10, scope: !4)
!16 = !DILocation(line: 2, column: 3, scope: !4)
