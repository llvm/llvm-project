; REQUIRES: asserts
; RUN: opt -passes='print<access-info>' -debug-only=loop-accesses \
; RUN:   -disable-output -S %s 2>&1 | FileCheck %s

define void @negative_step(ptr nocapture %A) {
; CHECK-LABEL: LAA: Checking a loop in 'negative_step' from negative_step.c:5:2
entry:
  %A.plus.1 = getelementptr i32, ptr %A, i64 1
  br label %loop

loop:
  %iv = phi i64 [ 1022, %entry ], [ %iv.next, %loop ]
  %gep.A = getelementptr inbounds i32, ptr %A, i64 %iv
  %l = load i32, ptr %gep.A, align 4
  %add = add nsw i32 %l, 1
  %gep.A.plus.1 = getelementptr i32, ptr %A.plus.1, i64 %iv
  store i32 %add, ptr %gep.A.plus.1, align 4
  %iv.next = add nsw i64 %iv, -1
  %cmp.not = icmp eq i64 %iv, 0
  br i1 %cmp.not, label %exit, label %loop, !dbg !2

exit:
  ret void
}

!llvm.module.flags = !{!5, !6, !7}

!0 = !DIFile(filename: "negative_step.c", directory: "/")
!1 = distinct !DISubprogram(name: "negative_step", scope: !0, file: !0, unit: !4)
!2 = !DILocation(line: 5, column: 2, scope: !1)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !0, producer: "clang")
!5 = !{i32 1, !"Debug Info Version", i32 3}
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 1, !"PIC Level", i32 2}
