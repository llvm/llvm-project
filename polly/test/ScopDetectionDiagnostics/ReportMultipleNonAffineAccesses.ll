; RUN: opt %loadPolly -basic-aa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-print-detect -disable-output < %s 2>&1| FileCheck %s
; RUN: opt %loadPolly -basic-aa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-print-detect -polly-delinearize=false -polly-detect-keep-going -disable-output < %s 2>&1| FileCheck %s -check-prefix=ALL
; RUN: opt %loadPolly -basic-aa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-print-detect -disable-output < %s 2>&1| FileCheck %s -check-prefix=DELIN
; RUN: opt %loadPolly -basic-aa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-print-detect -polly-detect-keep-going -disable-output < %s 2>&1| FileCheck %s -check-prefix=DELIN-ALL
; RUN: opt %loadPolly -basic-aa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-print-detect -polly-allow-nonaffine -disable-output < %s 2>&1| FileCheck %s -check-prefix=NONAFFINE
; RUN: opt %loadPolly -basic-aa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-print-detect -polly-allow-nonaffine -disable-output < %s 2>&1| FileCheck %s -check-prefix=NONAFFINE

;  1 void manyaccesses(float A[restrict], long n, float B[restrict][n])
;  2 {
;  3   for (long i = 0; i < 1024; ++i) {
;  4     float a1 = A[2 * i * i];
;  5     float a2 = A[2 * i * i + 1];
;  6     float b1 = B[0][0];
;  7     float b2 = B[i][i];
;  8     float b3 = B[i * i][i];
;  9     float b4 = B[i][0];
; 10     float b5 = B[0][i];
; 11     float b6 = B[0][i*i];
; 12
; 13     A[i * i] = a1 + a2 + b1 + b2 + b3 + b4 + b5 + b6;
; 14   }
; 15 }

; CHECK: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; CHECK-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; CHECK-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; ALL: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; ALL-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; ALL-NEXT: remark: /tmp/test.c:5:16: The array subscript of "A" is not affine
; -> B[0][0] is affine
; ALL-NEXT: remark: /tmp/test.c:7:16: The array subscript of "B" is not affine
; ALL-NEXT: remark: /tmp/test.c:8:16: The array subscript of "B" is not affine
; ALL-NEXT: remark: /tmp/test.c:9:16: The array subscript of "B" is not affine
; -> B[0][i] is affine
; ALL-NEXT: remark: /tmp/test.c:11:16: The array subscript of "B" is not affine
; ALL-NEXT: remark: /tmp/test.c:13:5: The array subscript of "A" is not affine
; ALL-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; DELIN: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; DELIN-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; DELIN-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; DELIN-ALL: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; DELIN-ALL-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; DELIN-ALL-NEXT: remark: /tmp/test.c:5:16: The array subscript of "A" is not affine
; DELIN-ALL-NEXT: remark: /tmp/test.c:13:5: The array subscript of "A" is not affine
; -> B[0][0] is affine if delinearized
; -> B[i][i] is affine if delinearized
; DELIN-ALL-NEXT: remark: /tmp/test.c:8:16: The array subscript of "B" is not affine
; -> B[i][0] is affine if delinearized
; -> B[0][i] is affine if delinearized
; DELIN-ALL-NEXT: remark: /tmp/test.c:11:16: The array subscript of "B" is not affine
; DELIN-ALL-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; NONAFFINE-NOT: remark: The following errors keep this region from being a Scop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @manyaccesses(ptr noalias %A, i64 %n, ptr noalias %B) !dbg !4 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp = add i64 %n, 1, !dbg !10
  br label %for.body, !dbg !10

for.body:                                         ; preds = %entry.split, %for.body
  %tmp3 = phi i64 [ 0, %entry.split ], [ %tmp14, %for.body ], !dbg !15
  %mul = mul i64 %tmp3, 2, !dbg !17
  %tmp4 = mul i64 %tmp, %tmp3, !dbg !18
  %arrayidx8 = getelementptr float, ptr %B, i64 %tmp4, !dbg !19
  %mul9 = mul i64 %n, %tmp3, !dbg !15
  %arrayidx12 = getelementptr float, ptr %B, i64 %mul9, !dbg !20
  %arrayidx15 = getelementptr float, ptr %B, i64 %tmp3, !dbg !21
  %mul1 = mul nsw i64 %mul, %tmp3, !dbg !17
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %mul1, !dbg !22
  %tmp5 = load float, ptr %arrayidx, align 4, !dbg !22
  %mul3 = mul nsw i64 %mul, %tmp3, !dbg !27
  %add1 = or i64 %mul3, 1, !dbg !27
  %arrayidx4 = getelementptr inbounds float, ptr %A, i64 %add1, !dbg !28
  %tmp6 = load float, ptr %arrayidx4, align 4, !dbg !28
  %tmp7 = load float, ptr %B, align 4, !dbg !29
  %tmp8 = load float, ptr %arrayidx8, align 4, !dbg !19
  %tmp9 = mul i64 %mul9, %tmp3, !dbg !15
  %arrayidx10.sum = add i64 %tmp9, %tmp3, !dbg !15
  %arrayidx11 = getelementptr inbounds float, ptr %B, i64 %arrayidx10.sum, !dbg !15
  %tmp10 = load float, ptr %arrayidx11, align 4, !dbg !15
  %tmp11 = load float, ptr %arrayidx12, align 4, !dbg !20
  %tmp12 = load float, ptr %arrayidx15, align 4, !dbg !21
  %mul16 = mul nsw i64 %tmp3, %tmp3, !dbg !30
  %arrayidx18 = getelementptr inbounds float, ptr %B, i64 %mul16, !dbg !31
  %tmp13 = load float, ptr %arrayidx18, align 4, !dbg !31
  %add19 = fadd float %tmp5, %tmp6, !dbg !32
  %add20 = fadd float %add19, %tmp7, !dbg !33
  %add21 = fadd float %add20, %tmp8, !dbg !34
  %add22 = fadd float %add21, %tmp10, !dbg !35
  %add23 = fadd float %add22, %tmp11, !dbg !36
  %add24 = fadd float %add23, %tmp12, !dbg !37
  %add25 = fadd float %add24, %tmp13, !dbg !38
  %mul26 = mul nsw i64 %tmp3, %tmp3, !dbg !39
  %arrayidx27 = getelementptr inbounds float, ptr %A, i64 %mul26, !dbg !40
  store float %add25, ptr %arrayidx27, align 4, !dbg !40
  %tmp14 = add nsw i64 %tmp3, 1, !dbg !41
  %exitcond = icmp ne i64 %tmp14, 1024, !dbg !10
  br i1 %exitcond, label %for.body, label %for.end, !dbg !10

for.end:                                          ; preds = %for.body
  ret void, !dbg !42
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: true, emissionKind: 2, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/home/grosser/Projects/polly/git/tools/polly/test")
!2 = !{}
!4 = distinct !DISubprogram(name: "manyaccesses", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "/tmp/test.c", directory: "/home/grosser/Projects/polly/git/tools/polly/test")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.6.0 "}
!10 = !DILocation(line: 3, column: 20, scope: !11)
!11 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !12)
!12 = !DILexicalBlockFile(discriminator: 1, file: !1, scope: !13)
!13 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !14)
!14 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !4)
!15 = !DILocation(line: 8, column: 16, scope: !16)
!16 = distinct !DILexicalBlock(line: 3, column: 35, file: !1, scope: !13)
!17 = !DILocation(line: 4, column: 26, scope: !16)
!18 = !DILocation(line: 4, column: 22, scope: !16)
!19 = !DILocation(line: 7, column: 16, scope: !16)
!20 = !DILocation(line: 9, column: 16, scope: !16)
!21 = !DILocation(line: 10, column: 16, scope: !16)
!22 = !DILocation(line: 4, column: 16, scope: !16)
!27 = !DILocation(line: 5, column: 26, scope: !16)
!28 = !DILocation(line: 5, column: 16, scope: !16)
!29 = !DILocation(line: 6, column: 16, scope: !16)
!30 = !DILocation(line: 11, column: 23, scope: !16) ; [ DW_TAG_lexical_block ] [/]
!31 = !DILocation(line: 11, column: 16, scope: !16) ; [ DW_TAG_lexical_block ] [/]
!32 = !DILocation(line: 13, column: 21, scope: !16)
!33 = !DILocation(line: 13, column: 26, scope: !16)
!34 = !DILocation(line: 13, column: 31, scope: !16)
!35 = !DILocation(line: 13, column: 36, scope: !16)
!36 = !DILocation(line: 13, column: 41, scope: !16)
!37 = !DILocation(line: 13, column: 46, scope: !16)
!38 = !DILocation(line: 13, column: 51, scope: !16)
!39 = !DILocation(line: 13, column: 11, scope: !16)
!40 = !DILocation(line: 13, column: 5, scope: !16)
!41 = !DILocation(line: 3, column: 30, scope: !13)
!42 = !DILocation(line: 15, column: 1, scope: !4)
