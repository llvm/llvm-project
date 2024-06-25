; REQUIRES: asserts
; RUN: opt -passes=loop-distribute -enable-loop-distribute \
; RUN:   -debug-only=loop-distribute -disable-output 2>&1 %s | FileCheck %s

define void @f(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %d, i64 %stride) {
; CHECK-LABEL: LDist: Checking a loop in 'f' from f.c:5:2
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %ind
  %load.a = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %ind
  %load.b = load i32, ptr %gep.b, align 4
  %mul.a = mul i32 %load.b, %load.a
  %add = add nuw nsw i64 %ind, 1
  %gep.a.plus4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mul.a, ptr %gep.a.plus4, align 4
  %gep.d = getelementptr inbounds i32, ptr %d, i64 %ind
  %loadD = load i32, ptr %gep.d, align 4
  %mul = mul i64 %ind, %stride
  %gep.strided.a = getelementptr inbounds i32, ptr %a, i64 %mul
  %load.strided.a = load i32, ptr %gep.strided.a, align 4
  %mul.c = mul i32 %loadD, %load.strided.a
  %gep.c = getelementptr inbounds i32, ptr %c, i64 %ind
  store i32 %mul.c, ptr %gep.c, align 4
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %exit, label %for.body, !dbg !2

exit:                                             ; preds = %for.body
  ret void
}

!llvm.module.flags = !{!5, !6, !7}

!0 = !DIFile(filename: "f.c", directory: "/")
!1 = distinct !DISubprogram(name: "f", scope: !0, file: !0, unit: !4)
!2 = !DILocation(line: 5, column: 2, scope: !1)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !0, producer: "clang")
!5 = !{i32 1, !"Debug Info Version", i32 3}
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 1, !"PIC Level", i32 2}
