; RUN: opt %s -csan -S | FileCheck %s
; RUN: opt %s -aa-pipeline=default -passes='cilksan' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [12 x i8] c"fib(%d)=%d\0A\00", align 1

; Function Attrs: nounwind readnone uwtable
define i32 @fib(i32 %n) local_unnamed_addr #0 !dbg !13 {
entry:
  %x = alloca i32, align 4
  %syncreg = tail call token @llvm.syncregion.start()
  call void @llvm.dbg.value(metadata i32 %n, metadata !17, metadata !DIExpression()), !dbg !19
  %cmp = icmp slt i32 %n, 2, !dbg !20
  br i1 %cmp, label %cleanup, label %if.end, !dbg !22

if.end:                                           ; preds = %entry
  %x.0.x.0..sroa_cast = bitcast i32* %x to i8*, !dbg !23
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %x.0.x.0..sroa_cast), !dbg !23
  detach within %syncreg, label %det.achd, label %det.cont, !dbg !24

det.achd:                                         ; preds = %if.end
  %sub = add nsw i32 %n, -1, !dbg !25
  %call = tail call i32 @fib(i32 %sub), !dbg !24
  call void @llvm.dbg.value(metadata i32 %call, metadata !18, metadata !DIExpression()), !dbg !26
  store i32 %call, i32* %x, align 4, !dbg !27
  reattach within %syncreg, label %det.cont, !dbg !27
; CHECK-LABEL: det.achd:
; CHECK: @__csan_store(
; CHECK-NEXT: store i32 %call, i32* %x

det.cont:                                         ; preds = %det.achd, %if.end
  %sub1 = add nsw i32 %n, -2, !dbg !28
  %call2 = tail call i32 @fib(i32 %sub1), !dbg !29
  %x.0.load10 = load i32, i32* %x, align 4, !dbg !30
  call void @llvm.dbg.value(metadata i32 %x.0.load10, metadata !18, metadata !DIExpression()), !dbg !26
  %add = add nsw i32 %x.0.load10, %call2, !dbg !30
  call void @llvm.dbg.value(metadata i32 %add, metadata !18, metadata !DIExpression()), !dbg !26
  store i32 %add, i32* %x, align 4, !dbg !30
  sync within %syncreg, label %sync.continue, !dbg !31
; CHECK-LABEL: det.cont:
; CHECK: @__csan_store(
; CHECK-NEXT: store i32 %add, i32* %x

sync.continue:                                    ; preds = %det.cont
  %x.0.load11 = load i32, i32* %x, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %x.0.load11, metadata !18, metadata !DIExpression()), !dbg !26
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %x.0.x.0..sroa_cast), !dbg !33
  br label %cleanup

cleanup:                                          ; preds = %entry, %sync.continue
  %retval.0 = phi i32 [ %x.0.load11, %sync.continue ], [ %n, %entry ]
  ret i32 %retval.0, !dbg !33
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #2 !dbg !34 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !38, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i8** %argv, metadata !39, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 10, metadata !40, metadata !DIExpression()), !dbg !44
  %cmp = icmp sgt i32 %argc, 1, !dbg !45
  br i1 %cmp, label %if.then, label %if.end, !dbg !47

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1, !dbg !48
  %0 = load i8*, i8** %arrayidx, align 8, !dbg !48, !tbaa !49
  %call = tail call i32 @atoi(i8* %0) #6, !dbg !53
  call void @llvm.dbg.value(metadata i32 %call, metadata !40, metadata !DIExpression()), !dbg !44
  br label %if.end, !dbg !54

if.end:                                           ; preds = %if.then, %entry
  %n.0 = phi i32 [ %call, %if.then ], [ 10, %entry ]
  call void @llvm.dbg.value(metadata i32 %n.0, metadata !40, metadata !DIExpression()), !dbg !44
  %call1 = tail call i32 @fib(i32 %n.0), !dbg !55
  call void @llvm.dbg.value(metadata i32 %call1, metadata !41, metadata !DIExpression()), !dbg !56
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), i32 %n.0, i32 %call1), !dbg !57
  ret i32 0, !dbg !58
; CHECK-LABEL: if.end:
; CHECK: @__csi_before_call(
; CHECK-NEXT: tail call i32 @fib(
; CHECK-NEXT: @__csi_after_call(
}

; Function Attrs: inlinehint nounwind readonly uwtable
define available_externally i32 @atoi(i8* nonnull %__nptr) local_unnamed_addr #3 !dbg !59 {
entry:
  call void @llvm.dbg.value(metadata i8* %__nptr, metadata !66, metadata !DIExpression()), !dbg !67
  %call = tail call i64 @strtol(i8* nocapture nonnull %__nptr, i8** null, i32 10) #7, !dbg !68
  %conv = trunc i64 %call to i32, !dbg !69
  ret i32 %conv, !dbg !70
}

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind
declare i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 1916cc670b59e15caed2d0a2ede0ae518b6a6a6c) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 3af5afd90a3d8cf4c84c782bde291871be5a2a95)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "fib_racy.c", directory: "/data/compilers/tests/cilksan")
!2 = !{}
!3 = !{!4, !5, !8}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 1916cc670b59e15caed2d0a2ede0ae518b6a6a6c) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 3af5afd90a3d8cf4c84c782bde291871be5a2a95)"}
!13 = distinct !DISubprogram(name: "fib", scope: !1, file: !1, line: 5, type: !14, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!4, !4}
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "n", arg: 1, scope: !13, file: !1, line: 5, type: !4)
!18 = !DILocalVariable(name: "x", scope: !13, file: !1, line: 7, type: !4)
!19 = !DILocation(line: 5, column: 13, scope: !13)
!20 = !DILocation(line: 6, column: 9, scope: !21)
!21 = distinct !DILexicalBlock(scope: !13, file: !1, line: 6, column: 7)
!22 = !DILocation(line: 6, column: 7, scope: !13)
!23 = !DILocation(line: 7, column: 3, scope: !13)
!24 = !DILocation(line: 8, column: 18, scope: !13)
!25 = !DILocation(line: 8, column: 23, scope: !13)
!26 = !DILocation(line: 7, column: 7, scope: !13)
!27 = !DILocation(line: 8, column: 5, scope: !13)
!28 = !DILocation(line: 9, column: 13, scope: !13)
!29 = !DILocation(line: 9, column: 8, scope: !13)
!30 = !DILocation(line: 9, column: 5, scope: !13)
!31 = !DILocation(line: 10, column: 3, scope: !13)
!32 = !DILocation(line: 11, column: 10, scope: !13)
!33 = !DILocation(line: 12, column: 1, scope: !13)
!34 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !35, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !37)
!35 = !DISubroutineType(types: !36)
!36 = !{!4, !4, !5}
!37 = !{!38, !39, !40, !41}
!38 = !DILocalVariable(name: "argc", arg: 1, scope: !34, file: !1, line: 14, type: !4)
!39 = !DILocalVariable(name: "argv", arg: 2, scope: !34, file: !1, line: 14, type: !5)
!40 = !DILocalVariable(name: "n", scope: !34, file: !1, line: 15, type: !4)
!41 = !DILocalVariable(name: "result", scope: !34, file: !1, line: 19, type: !4)
!42 = !DILocation(line: 14, column: 14, scope: !34)
!43 = !DILocation(line: 14, column: 26, scope: !34)
!44 = !DILocation(line: 15, column: 7, scope: !34)
!45 = !DILocation(line: 16, column: 12, scope: !46)
!46 = distinct !DILexicalBlock(scope: !34, file: !1, line: 16, column: 7)
!47 = !DILocation(line: 16, column: 7, scope: !34)
!48 = !DILocation(line: 17, column: 14, scope: !46)
!49 = !{!50, !50, i64 0}
!50 = !{!"any pointer", !51, i64 0}
!51 = !{!"omnipotent char", !52, i64 0}
!52 = !{!"Simple C/C++ TBAA"}
!53 = !DILocation(line: 17, column: 9, scope: !46)
!54 = !DILocation(line: 17, column: 5, scope: !46)
!55 = !DILocation(line: 19, column: 16, scope: !34)
!56 = !DILocation(line: 19, column: 7, scope: !34)
!57 = !DILocation(line: 20, column: 3, scope: !34)
!58 = !DILocation(line: 22, column: 1, scope: !34)
!59 = distinct !DISubprogram(name: "atoi", scope: !60, file: !60, line: 361, type: !61, isLocal: false, isDefinition: true, scopeLine: 362, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !65)
!60 = !DIFile(filename: "/usr/include/stdlib.h", directory: "/data/compilers/tests/cilksan")
!61 = !DISubroutineType(types: !62)
!62 = !{!4, !63}
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !64, size: 64)
!64 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!65 = !{!66}
!66 = !DILocalVariable(name: "__nptr", arg: 1, scope: !59, file: !60, line: 361, type: !63)
!67 = !DILocation(line: 361, column: 1, scope: !59)
!68 = !DILocation(line: 363, column: 16, scope: !59)
!69 = !DILocation(line: 363, column: 10, scope: !59)
!70 = !DILocation(line: 364, column: 1, scope: !59)
