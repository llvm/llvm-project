; RUN: llc -relocation-model=static -arm-promote-constant < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m--linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"abc\00", align 1

; CHECK-LABEL: fn1
; CHECK: .long .L.str
define arm_aapcscc ptr @fn1() local_unnamed_addr #0 !dbg !8 {
entry:
  ret ptr @.str, !dbg !14
}

; CHECK-LABEL: fn2
; CHECK: .long .L.str
define arm_aapcscc ptr @fn2() local_unnamed_addr #0 !dbg !15 {
entry:
  ret ptr getelementptr inbounds ([4 x i8], ptr @.str, i32 0, i32 1), !dbg !16
}

attributes #0 = { minsize norecurse nounwind optsize readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-m3" "target-features"="+hwdiv,+soft-float,-crypto,-neon" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/Users/jammol01/Code/test.c", directory: "/Users/jammol01/Code/llvm-git/build")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 3.9.0"}
!8 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 32, align: 32)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!14 = !DILocation(line: 2, column: 5, scope: !8)
!15 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 4, type: !9, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 5, column: 5, scope: !15)
