; RUN: opt < %s -add-discriminators -sroa -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test case obtained from the following C code:

; struct A {
;  int field1;
;  short field2;
; };
;
; struct B {
;   struct A field1;
;   int field2;
; };
;
;
; extern struct B g_b;
; extern int bar(struct B b, int c);
;
; int foo(int cond) {
;   int result = cond ? bar(g_b, 33) : 42;
;   return result;
; }

; In this test, global variable g_b is passed by copy to function bar. That
; copy is located on the stack (see alloca %g_b.coerce), and it is initialized
; by a memcpy call.
;
; SROA would split alloca %g_b.coerce into two (smaller disjoint) slices:
; slice [0,8) and slice [8, 12). Users of the original alloca are rewritten
; as users of the new alloca slices.
; In particular, the memcpy is rewritten by SROA as two load/store pairs.
;
; Later on, mem2reg successfully promotes the new alloca slices to registers,
; and loads %3 and %5 are made redundant by the loads obtained from the memcpy
; intrinsic expansion.
;
; If pass AddDiscriminators doesn't assign a discriminator to the intrinsic
; memcpy call, then the loads obtained from the memcpy expansion would not have
; a correct discriminator.
;
; This test checks that the two new loads inserted by SROA in %cond.true
; correctly reference a debug location with a non-zero discriminator. This test
; also checks that the same discriminator is used by all instructions from
; basic block %cond.true.

%struct.B = type { %struct.A, i32 }
%struct.A = type { i32, i16 }

@g_b = external global %struct.B, align 4

define i32 @foo(i32 %cond) #0 !dbg !5 {
entry:
  %g_b.coerce = alloca { i64, i32 }, align 4
  %tobool = icmp ne i32 %cond, 0, !dbg !7
  br i1 %tobool, label %cond.true, label %cond.end, !dbg !7

cond.true:
; CHECK-LABEL: cond.true:
; CHECK:       load i64, {{.*}}, !dbg ![[LOC:[0-9]+]]
; CHECK-NEXT:  load i32, {{.*}}, !dbg ![[LOC]]
; CHECK-NEXT:  %call = call i32 @bar({{.*}}), !dbg ![[LOC]]
; CHECK-NEXT:  br label %cond.end, !dbg ![[BR_LOC:[0-9]+]]

; CHECK-DAG: ![[LOC]] = !DILocation(line: 16, column: 23, scope: ![[SCOPE:[0-9]+]])
; CHECK-DAG: ![[SCOPE]] = !DILexicalBlockFile({{.*}}, discriminator: 2)
; CHECK-DAG: ![[BR_LOC]] = !DILocation(line: 16, column: 16, scope: ![[SCOPE]])

  %0 = bitcast { i64, i32 }* %g_b.coerce to i8*, !dbg !8
  %1 = bitcast %struct.B* @g_b to i8*, !dbg !8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 12, i32 4, i1 false), !dbg !8
  %2 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %g_b.coerce, i32 0, i32 0, !dbg !8
  %3 = load i64, i64* %2, align 4, !dbg !8
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %g_b.coerce, i32 0, i32 1, !dbg !8
  %5 = load i32, i32* %4, align 4, !dbg !8
  %call = call i32 @bar(i64 %3, i32 %5, i32 33), !dbg !8
  br label %cond.end, !dbg !7

cond.end:                                         ; preds = %entry, %cond.true
  %cond1 = phi i32 [ %call, %cond.true ], [ 42, %entry ], !dbg !7
  ret i32 %cond1, !dbg !9
}

declare i32 @bar(i64, i32, i32)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

attributes #0 = { noinline nounwind uwtable }
attributes #1 = { argmemonly nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 15, type: !6, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 16, column: 16, scope: !5)
!8 = !DILocation(line: 16, column: 23, scope: !5)
!9 = !DILocation(line: 17, column: 3, scope: !5)
