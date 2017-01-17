; RUN: not llc -march=bpfel < %s 2>&1 >/dev/null | FileCheck %s

;; CHECK-NOT: nowarn
define void @nowarn() local_unnamed_addr #0 !dbg !6 {
  %1 = alloca [504 x i8], align 1
  %2 = getelementptr inbounds [504 x i8], [504 x i8]* %1, i64 0, i64 0, !dbg !15
  call void @llvm.lifetime.start(i64 504, i8* nonnull %2) #4, !dbg !15
  tail call void @llvm.dbg.declare(metadata [504 x i8]* %1, metadata !10, metadata !16), !dbg !17
  call void @doit(i8* nonnull %2) #4, !dbg !18
  call void @llvm.lifetime.end(i64 504, i8* nonnull %2) #4, !dbg !19
  ret void, !dbg !19
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare void @doit(i8*) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

; CHECK: error: warn_stack.c
; CHECK: BPF stack limit
define void @warn() local_unnamed_addr #0 !dbg !20 {
  %1 = alloca [512 x i8], align 1
  %2 = getelementptr inbounds [512 x i8], [512 x i8]* %1, i64 0, i64 0, !dbg !26
  call void @llvm.lifetime.start(i64 512, i8* nonnull %2) #4, !dbg !26
  tail call void @llvm.dbg.declare(metadata [512 x i8]* %1, metadata !22, metadata !16), !dbg !27
  call void @doit(i8* nonnull %2) #4, !dbg !28
  call void @llvm.lifetime.end(i64 512, i8* nonnull %2) #4, !dbg !29
  ret void, !dbg !29
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 292141) (llvm/trunk 292156)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "warn_stack.c", directory: "/w/llvm/bld")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0 (trunk 292141) (llvm/trunk 292156)"}
!6 = distinct !DISubprogram(name: "nowarn", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10}
!10 = !DILocalVariable(name: "buf", scope: !6, file: !1, line: 4, type: !11)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 4088, elements: !13)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = !{!14}
!14 = !DISubrange(count: 504)
!15 = !DILocation(line: 4, column: 2, scope: !6)
!16 = !DIExpression()
!17 = !DILocation(line: 4, column: 7, scope: !6)
!18 = !DILocation(line: 5, column: 2, scope: !6)
!19 = !DILocation(line: 6, column: 1, scope: !6)
!20 = distinct !DISubprogram(name: "warn", scope: !1, file: !1, line: 7, type: !7, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !21)
!21 = !{!22}
!22 = !DILocalVariable(name: "buf", scope: !20, file: !1, line: 9, type: !23)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 4096, elements: !24)
!24 = !{!25}
!25 = !DISubrange(count: 512)
!26 = !DILocation(line: 9, column: 2, scope: !20)
!27 = !DILocation(line: 9, column: 7, scope: !20)
!28 = !DILocation(line: 10, column: 2, scope: !20)
!29 = !DILocation(line: 11, column: 1, scope: !20)
