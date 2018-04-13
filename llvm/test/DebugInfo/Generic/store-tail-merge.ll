; RUN: opt -instcombine -S < %s | FileCheck %s
;
; Generated with:
;
; clang -S -gmlt -emit-llvm test.c -o 1.ll
; opt -sroa -S 1.ll -o test.ll
;
; extern int bar(int i);
; extern int bar2(int i);
;
; int foo(int a, int *d) {
;   if(a) {
;       *d = bar(a);
;   } else {
;       *d = bar2(a);
;   }
;
;   return a;
; }
;
; CHECK:       define {{.*}}@foo
; CHECK:       if.end:
; CHECK-NEXT:  %storemerge = phi
;
; The debug location on the store should be a line-0 location.
; CHECK-NEXT:  store i32 %storemerge{{.*}}, align 4, !dbg [[storeLoc:![0-9]+]]
; CHECK: [[storeLoc]] = !DILocation(line: 0
;
; ModuleID = 'test1.ll'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @foo(i32 %a, i32* %d) !dbg !6 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !8
  br i1 %tobool, label %if.then, label %if.else, !dbg !8

if.then:                                          ; preds = %entry
  %call = call i32 @bar(i32 %a), !dbg !9
  store i32 %call, i32* %d, align 4, !dbg !10
  br label %if.end, !dbg !11

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar2(i32 %a), !dbg !12
  store i32 %call1, i32* %d, align 4, !dbg !13
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 %a, !dbg !14
}

declare i32 @bar(i32)

declare i32 @bar2(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/probinson/projects/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 5, column: 6, scope: !6)
!9 = !DILocation(line: 6, column: 12, scope: !6)
!10 = !DILocation(line: 6, column: 10, scope: !6)
!11 = !DILocation(line: 7, column: 3, scope: !6)
!12 = !DILocation(line: 8, column: 12, scope: !6)
!13 = !DILocation(line: 8, column: 10, scope: !6)
!14 = !DILocation(line: 10, column: 3, scope: !6)
