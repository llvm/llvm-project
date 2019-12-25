; Test that alloca merging in the inliner places dbg.declare calls immediately
; after the merged alloca. Not at the end of the entry BB, and definitely not
; before the alloca.
;
; clang -g -S -emit-llvm -Xclang -disable-llvm-optzns
;
;__attribute__((always_inline)) void f() {
;  char aaa[100];
;  aaa[10] = 1;
;}
;
;__attribute__((always_inline)) void g() {
;  char bbb[100];
;  bbb[20] = 1;
;}
;
;void h() {
;  f();
;  g();
;}
;
; RUN: opt -always-inline -S < %s | FileCheck %s
;
; CHECK:      define void @h()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[AI:.*]] = alloca [100 x i8]
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata [100 x i8]* %[[AI]],
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata [100 x i8]* %[[AI]],


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline nounwind uwtable
define void @f() #0 !dbg !4 {
entry:
  %aaa = alloca [100 x i8], align 16
  call void @llvm.dbg.declare(metadata [100 x i8]* %aaa, metadata !12, metadata !17), !dbg !18
  %arrayidx = getelementptr inbounds [100 x i8], [100 x i8]* %aaa, i64 0, i64 10, !dbg !19
  store i8 1, i8* %arrayidx, align 2, !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: alwaysinline nounwind uwtable
define void @g() #0 !dbg !7 {
entry:
  %bbb = alloca [100 x i8], align 16
  call void @llvm.dbg.declare(metadata [100 x i8]* %bbb, metadata !22, metadata !17), !dbg !23
  %arrayidx = getelementptr inbounds [100 x i8], [100 x i8]* %bbb, i64 0, i64 20, !dbg !24
  store i8 1, i8* %arrayidx, align 4, !dbg !25
  ret void, !dbg !26
}

; Function Attrs: nounwind uwtable
define void @h() #2 !dbg !8 {
entry:
  call void @f(), !dbg !27
  call void @g(), !dbg !28
  ret void, !dbg !29
}

attributes #0 = { alwaysinline nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 248518) (llvm/trunk 248512)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "../1.c", directory: "/code/llvm-git/build")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 6, type: !5, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 11, type: !5, isLocal: false, isDefinition: true, scopeLine: 11, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.8.0 (trunk 248518) (llvm/trunk 248512)"}
!12 = !DILocalVariable(name: "aaa", scope: !4, file: !1, line: 2, type: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 800, align: 8, elements: !15)
!14 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!15 = !{!16}
!16 = !DISubrange(count: 100)
!17 = !DIExpression()
!18 = !DILocation(line: 2, column: 8, scope: !4)
!19 = !DILocation(line: 3, column: 3, scope: !4)
!20 = !DILocation(line: 3, column: 11, scope: !4)
!21 = !DILocation(line: 4, column: 1, scope: !4)
!22 = !DILocalVariable(name: "bbb", scope: !7, file: !1, line: 7, type: !13)
!23 = !DILocation(line: 7, column: 8, scope: !7)
!24 = !DILocation(line: 8, column: 3, scope: !7)
!25 = !DILocation(line: 8, column: 11, scope: !7)
!26 = !DILocation(line: 9, column: 1, scope: !7)
!27 = !DILocation(line: 12, column: 3, scope: !8)
!28 = !DILocation(line: 13, column: 3, scope: !8)
!29 = !DILocation(line: 14, column: 1, scope: !8)
