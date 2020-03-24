; RUN: opt %s -csan -S | FileCheck %s
; RUN: opt %s -aa-pipeline=default -passes='cilksan' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [12 x i8] c"fib(%d)=%d\0A\00", align 1

; Function Attrs: nounwind readnone sanitize_cilk uwtable
define i32 @fib(i32 %n) local_unnamed_addr #0 !dbg !13 {
entry:
  %x = alloca i32, align 4
  %syncreg = tail call token @llvm.syncregion.start()
  call void @llvm.dbg.value(metadata i32 %n, metadata !17, metadata !DIExpression()), !dbg !20
  %cmp = icmp slt i32 %n, 2, !dbg !21
  br i1 %cmp, label %return, label %if.end, !dbg !23

if.end:                                           ; preds = %entry
  %x.0.x.0..sroa_cast = bitcast i32* %x to i8*, !dbg !24
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %x.0.x.0..sroa_cast), !dbg !24
  detach within %syncreg, label %det.achd, label %det.cont, !dbg !25

det.achd:                                         ; preds = %if.end
  %sub = add nsw i32 %n, -1, !dbg !26
  %call = tail call i32 @fib(i32 %sub), !dbg !25
  call void @llvm.dbg.value(metadata i32 %call, metadata !18, metadata !DIExpression()), !dbg !27
  store i32 %call, i32* %x, align 4, !dbg !28
  reattach within %syncreg, label %det.cont, !dbg !28
; CHECK-LABEL: det.achd:
; CHECK-NOT: @__csan_store(
; CHECK: store i32 %call, i32* %x

det.cont:                                         ; preds = %det.achd, %if.end
  %sub1 = add nsw i32 %n, -2, !dbg !29
  %call2 = tail call i32 @fib(i32 %sub1), !dbg !30
  call void @llvm.dbg.value(metadata i32 %call2, metadata !19, metadata !DIExpression()), !dbg !31
  sync within %syncreg, label %sync.continue, !dbg !32

sync.continue:                                    ; preds = %det.cont
  %x.0.load9 = load i32, i32* %x, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %x.0.load9, metadata !18, metadata !DIExpression()), !dbg !27
  %add = add nsw i32 %x.0.load9, %call2, !dbg !34
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %x.0.x.0..sroa_cast), !dbg !35
  br label %return
; CHECK-LABEL: sync.continue:
; CHECK-NOT: @__csan_load(
; CHECK: load i32, i32* %x

return:                                           ; preds = %entry, %sync.continue
  %retval.0 = phi i32 [ %add, %sync.continue ], [ %n, %entry ]
  ret i32 %retval.0, !dbg !35
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind sanitize_cilk uwtable
define i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #2 !dbg !36 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !40, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i8** %argv, metadata !41, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 10, metadata !42, metadata !DIExpression()), !dbg !46
  %cmp = icmp sgt i32 %argc, 1, !dbg !47
  br i1 %cmp, label %if.then, label %if.end, !dbg !49

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1, !dbg !50
  %0 = load i8*, i8** %arrayidx, align 8, !dbg !50, !tbaa !51
  %call = tail call i32 @atoi(i8* %0) #6, !dbg !55
  call void @llvm.dbg.value(metadata i32 %call, metadata !42, metadata !DIExpression()), !dbg !46
  br label %if.end, !dbg !56

if.end:                                           ; preds = %if.then, %entry
  %n.0 = phi i32 [ %call, %if.then ], [ 10, %entry ]
  call void @llvm.dbg.value(metadata i32 %n.0, metadata !42, metadata !DIExpression()), !dbg !46
  %call1 = tail call i32 @fib(i32 %n.0), !dbg !57
  call void @llvm.dbg.value(metadata i32 %call1, metadata !43, metadata !DIExpression()), !dbg !58
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), i32 %n.0, i32 %call1), !dbg !59
  ret i32 0, !dbg !60
; CHECK-LABEL: if.end:
; CHECK-NOT: call void @__cilksan_disable_checking(
; CHECK: tail call i32 @fib(
; CHECK-NOT: call void @__cilksan_enable_checking(
}

; Function Attrs: inlinehint nounwind readonly uwtable
define available_externally i32 @atoi(i8* nonnull %__nptr) local_unnamed_addr #3 !dbg !61 {
entry:
  call void @llvm.dbg.value(metadata i8* %__nptr, metadata !68, metadata !DIExpression()), !dbg !69
  %call = tail call i64 @strtol(i8* nocapture nonnull %__nptr, i8** null, i32 10) #7, !dbg !70
  %conv = trunc i64 %call to i32, !dbg !71
  ret i32 %conv, !dbg !72
}

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind
declare i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { nounwind readnone sanitize_cilk uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind sanitize_cilk uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 051bd478f93bf64db3934d14f97a36137bffef5e) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 9de43afffece94ca0534b391544bbfd246fc7b91)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "fib.c", directory: "/data/compilers/tests/cilksan")
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
!12 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 051bd478f93bf64db3934d14f97a36137bffef5e) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 9de43afffece94ca0534b391544bbfd246fc7b91)"}
!13 = distinct !DISubprogram(name: "fib", scope: !1, file: !1, line: 5, type: !14, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!4, !4}
!16 = !{!17, !18, !19}
!17 = !DILocalVariable(name: "n", arg: 1, scope: !13, file: !1, line: 5, type: !4)
!18 = !DILocalVariable(name: "x", scope: !13, file: !1, line: 7, type: !4)
!19 = !DILocalVariable(name: "y", scope: !13, file: !1, line: 7, type: !4)
!20 = !DILocation(line: 5, column: 13, scope: !13)
!21 = !DILocation(line: 6, column: 9, scope: !22)
!22 = distinct !DILexicalBlock(scope: !13, file: !1, line: 6, column: 7)
!23 = !DILocation(line: 6, column: 7, scope: !13)
!24 = !DILocation(line: 7, column: 3, scope: !13)
!25 = !DILocation(line: 8, column: 18, scope: !13)
!26 = !DILocation(line: 8, column: 23, scope: !13)
!27 = !DILocation(line: 7, column: 7, scope: !13)
!28 = !DILocation(line: 8, column: 5, scope: !13)
!29 = !DILocation(line: 9, column: 12, scope: !13)
!30 = !DILocation(line: 9, column: 7, scope: !13)
!31 = !DILocation(line: 7, column: 10, scope: !13)
!32 = !DILocation(line: 10, column: 3, scope: !13)
!33 = !DILocation(line: 11, column: 10, scope: !13)
!34 = !DILocation(line: 11, column: 11, scope: !13)
!35 = !DILocation(line: 12, column: 1, scope: !13)
!36 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !37, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !39)
!37 = !DISubroutineType(types: !38)
!38 = !{!4, !4, !5}
!39 = !{!40, !41, !42, !43}
!40 = !DILocalVariable(name: "argc", arg: 1, scope: !36, file: !1, line: 14, type: !4)
!41 = !DILocalVariable(name: "argv", arg: 2, scope: !36, file: !1, line: 14, type: !5)
!42 = !DILocalVariable(name: "n", scope: !36, file: !1, line: 15, type: !4)
!43 = !DILocalVariable(name: "result", scope: !36, file: !1, line: 19, type: !4)
!44 = !DILocation(line: 14, column: 14, scope: !36)
!45 = !DILocation(line: 14, column: 26, scope: !36)
!46 = !DILocation(line: 15, column: 7, scope: !36)
!47 = !DILocation(line: 16, column: 12, scope: !48)
!48 = distinct !DILexicalBlock(scope: !36, file: !1, line: 16, column: 7)
!49 = !DILocation(line: 16, column: 7, scope: !36)
!50 = !DILocation(line: 17, column: 14, scope: !48)
!51 = !{!52, !52, i64 0}
!52 = !{!"any pointer", !53, i64 0}
!53 = !{!"omnipotent char", !54, i64 0}
!54 = !{!"Simple C/C++ TBAA"}
!55 = !DILocation(line: 17, column: 9, scope: !48)
!56 = !DILocation(line: 17, column: 5, scope: !48)
!57 = !DILocation(line: 19, column: 16, scope: !36)
!58 = !DILocation(line: 19, column: 7, scope: !36)
!59 = !DILocation(line: 20, column: 3, scope: !36)
!60 = !DILocation(line: 21, column: 3, scope: !36)
!61 = distinct !DISubprogram(name: "atoi", scope: !62, file: !62, line: 361, type: !63, isLocal: false, isDefinition: true, scopeLine: 362, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !67)
!62 = !DIFile(filename: "/usr/include/stdlib.h", directory: "/data/compilers/tests/cilksan")
!63 = !DISubroutineType(types: !64)
!64 = !{!4, !65}
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !66, size: 64)
!66 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!67 = !{!68}
!68 = !DILocalVariable(name: "__nptr", arg: 1, scope: !61, file: !62, line: 361, type: !65)
!69 = !DILocation(line: 361, column: 1, scope: !61)
!70 = !DILocation(line: 363, column: 16, scope: !61)
!71 = !DILocation(line: 363, column: 10, scope: !61)
!72 = !DILocation(line: 363, column: 3, scope: !61)
