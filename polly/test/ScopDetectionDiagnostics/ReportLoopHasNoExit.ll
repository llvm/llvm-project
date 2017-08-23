; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-allow-nonaffine-loops -analyze  -polly-detect < %s 2>&1 | FileCheck %s
; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-allow-nonaffine-loops=false -analyze  -polly-detect < %s 2>&1 | FileCheck %s

; void func (int param0, int N, int *A)
; {
;   for (int i = 0; i < N; i++)
;     if (param0)
;       while (1)
;         A[i] = 1;
;     else
;      A[i] = 2;
; }

; CHECK: remark: ReportLoopHasNoExit.c:7:7: Loop cannot be handled because it has no exit.



target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @func(i32 %param0, i32 %N, i32* %A) #0 !dbg !6 {
entry:
  %param0.addr = alloca i32, align 4
  %N.addr = alloca i32, align 4
  %A.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store i32 %param0, i32* %param0.addr, align 4
  store i32 %N, i32* %N.addr, align 4
  store i32* %A, i32** %A.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %N.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end, !dbg !27

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %param0.addr, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  br label %while.body

while.body:                                       ; preds = %if.then, %while.body
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %4 = load i32*, i32** %A.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %4, i64 %idxprom
  store i32 1, i32* %arrayidx, align 4
  br label %while.body, !dbg !37

if.else:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %5 to i64
  %6 = load i32*, i32** %A.addr, align 8
  %arrayidx2 = getelementptr inbounds i32, i32* %6, i64 %idxprom1
  store i32 2, i32* %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.else
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "ReportLoopHasNoExit.c", directory: "test/ScopDetectionDiagnostics/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1,  isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!19 = distinct !DILexicalBlock(scope: !6, file: !1, line: 3, column: 3)
!23 = !DILexicalBlockFile(scope: !24, file: !1, discriminator: 1)
!24 = distinct !DILexicalBlock(scope: !19, file: !1, line: 3, column: 3)
!27 = !DILocation(line: 3, column: 3, scope: !23)
!29 = distinct !DILexicalBlock(scope: !30, file: !1, line: 5, column: 9)
!30 = distinct !DILexicalBlock(scope: !24, file: !1, line: 4, column: 3)
!33 = distinct !DILexicalBlock(scope: !29, file: !1, line: 6, column: 5)
!37 = !DILocation(line: 7, column: 7, scope: !38)
!38 = !DILexicalBlockFile(scope: !33, file: !1, discriminator: 1)
