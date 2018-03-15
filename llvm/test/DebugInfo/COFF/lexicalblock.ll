; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
;
; -- lexicablock.cxx begin ----------------------------------------------------
; int main(int argc, char *argv[]) {
;   int localA = 1;
;
;   { // S_BLOCK32 not emitted because it has multiple address ranges.
;     int localB = 2;
;
;     if (__builtin_expect(argc != 1, 0)) { // S_BLOCK32 containing 'localC'
;       int localC = 3;
;     }
;   }
;
;   { // S_BLOCK32 containing 'localD'
;     int localD = 4;
;     localA = localD;
;   }
;
;   { // S_BLOCK32 not emitted
;     { // S_BLOCK32 containing 'localE'
;       int localE = 5;
;       localA = localE;
;     }
;   }
;
;   { // S_BLOCK32 containing 'localF'
;     int localF = 6;
;     localA = localF;
;
;     { // S_BLOCK32 containing 'localG'
;       int localG = 7;
;       localA = localG;
;     }
;   }
;
;   if (localA == 7) { // S_BLOCK32 containing 'localH'
;     int localH = 8;
;     localA = localH;
;   }
;
;   return localA != 8 ? -1 : 0;
; }
; -- lexicalblock.cxx end -----------------------------------------------------
;
; To regenerate the IR below:
;   $ clang -cc1 -triple i686-pc-windows -emit-llvm -o lexicalblock.tmp -debug-info-kind=limited -gcodeview lexicablock.cxx -O1 -disable-llvm-passes
;   $ opt -lower-expect -S -o lexicalblock.ll < lexicalblock.tmp
;
; The commands above split the lexical block containing localB and localC into
; two parts, thus creating multiple ranges for the containing lexical block
; without optimizing out the whole thing.
;
; CHECK: {{.*}}Proc{{.*}}Sym {
; CHECK:   DisplayName: main
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: argc
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: argv
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localA
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localB
; CHECK: }
; CHECK: BlockSym {
; CHECK:   Kind: S_BLOCK32 {{.*}}
; CHECK:   BlockName: 
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localC
; CHECK: }
; CHECK: ScopeEndSym {
; CHECK:   Kind: S_END {{.*}}
; CHECK: }
; CHECK: BlockSym {
; CHECK:   Kind: S_BLOCK32 {{.*}}
; CHECK:   BlockName: 
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localD
; CHECK: }
; CHECK: ScopeEndSym {
; CHECK:   Kind: S_END {{.*}}
; CHECK: }
; CHECK: BlockSym {
; CHECK:   Kind: S_BLOCK32 {{.*}}
; CHECK:   BlockName: 
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localE
; CHECK: }
; CHECK: ScopeEndSym {
; CHECK: }
; CHECK: BlockSym {
; CHECK:   Kind: S_BLOCK32 {{.*}}
; CHECK:   BlockName: 
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localF
; CHECK: }
; CHECK: BlockSym {
; CHECK:   Kind: S_BLOCK32 {{.*}}
; CHECK:   BlockName: 
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localG
; CHECK: }
; CHECK: ScopeEndSym {
; CHECK:   Kind: S_END {{.*}}
; CHECK: }
; CHECK: ScopeEndSym {
; CHECK:   Kind: S_END {{.*}}
; CHECK: }
; CHECK: BlockSym {
; CHECK:   Kind: S_BLOCK32 {{.*}}
; CHECK:   BlockName: 
; CHECK: }
; CHECK: LocalSym {
; CHECK:   VarName: localH
; CHECK: }
; CHECK: ScopeEndSym {
; CHECK:   Kind: S_END {{.*}}
; CHECK: }
; CHECK: ProcEnd {
; CHECK: }
;
; ModuleID = 'lexicalblock.cxx'
source_filename = "lexicalblock.cxx"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

; Function Attrs: norecurse nounwind
define i32 @main(i32 %argc, i8** %argv) #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %argv.addr = alloca i8**, align 4
  %argc.addr = alloca i32, align 4
  %localA = alloca i32, align 4
  %localB = alloca i32, align 4
  %localC = alloca i32, align 4
  %localD = alloca i32, align 4
  %localE = alloca i32, align 4
  %localF = alloca i32, align 4
  %localG = alloca i32, align 4
  %localH = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i8** %argv, i8*** %argv.addr, align 4, !tbaa !37
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !17, metadata !DIExpression()), !dbg !41
  store i32 %argc, i32* %argc.addr, align 4, !tbaa !42
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !18, metadata !DIExpression()), !dbg !41
  %0 = bitcast i32* %localA to i8*, !dbg !44
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4, !dbg !44
  call void @llvm.dbg.declare(metadata i32* %localA, metadata !19, metadata !DIExpression()), !dbg !44
  store i32 1, i32* %localA, align 4, !dbg !44, !tbaa !42
  %1 = bitcast i32* %localB to i8*, !dbg !45
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %localB, metadata !20, metadata !DIExpression()), !dbg !45
  store i32 2, i32* %localB, align 4, !dbg !45, !tbaa !42
  %2 = load i32, i32* %argc.addr, align 4, !dbg !46, !tbaa !42
  %cmp = icmp ne i32 %2, 1, !dbg !46
  %conv = zext i1 %cmp to i32, !dbg !46
  %tobool = icmp ne i32 %conv, 0, !dbg !46
  br i1 %tobool, label %if.then, label %if.end, !dbg !46, !prof !47

if.then:                                          ; preds = %entry
  %3 = bitcast i32* %localC to i8*, !dbg !48
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %localC, metadata !22, metadata !DIExpression()), !dbg !48
  store i32 3, i32* %localC, align 4, !dbg !48, !tbaa !42
  %4 = bitcast i32* %localC to i8*, !dbg !49
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #4, !dbg !49
  br label %if.end, !dbg !49

if.end:                                           ; preds = %if.then, %entry
  %5 = bitcast i32* %localB to i8*, !dbg !50
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #4, !dbg !50
  %6 = bitcast i32* %localD to i8*, !dbg !51
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4, !dbg !51
  call void @llvm.dbg.declare(metadata i32* %localD, metadata !25, metadata !DIExpression()), !dbg !51
  store i32 4, i32* %localD, align 4, !dbg !51, !tbaa !42
  %7 = load i32, i32* %localD, align 4, !dbg !52, !tbaa !42
  store i32 %7, i32* %localA, align 4, !dbg !52, !tbaa !42
  %8 = bitcast i32* %localD to i8*, !dbg !53
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %8) #4, !dbg !53
  %9 = bitcast i32* %localE to i8*, !dbg !54
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #4, !dbg !54
  call void @llvm.dbg.declare(metadata i32* %localE, metadata !27, metadata !DIExpression()), !dbg !54
  store i32 5, i32* %localE, align 4, !dbg !54, !tbaa !42
  %10 = load i32, i32* %localE, align 4, !dbg !55, !tbaa !42
  store i32 %10, i32* %localA, align 4, !dbg !55, !tbaa !42
  %11 = bitcast i32* %localE to i8*, !dbg !56
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %11) #4, !dbg !56
  %12 = bitcast i32* %localF to i8*, !dbg !57
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %12) #4, !dbg !57
  call void @llvm.dbg.declare(metadata i32* %localF, metadata !30, metadata !DIExpression()), !dbg !57
  store i32 6, i32* %localF, align 4, !dbg !57, !tbaa !42
  %13 = load i32, i32* %localF, align 4, !dbg !58, !tbaa !42
  store i32 %13, i32* %localA, align 4, !dbg !58, !tbaa !42
  %14 = bitcast i32* %localG to i8*, !dbg !59
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %14) #4, !dbg !59
  call void @llvm.dbg.declare(metadata i32* %localG, metadata !32, metadata !DIExpression()), !dbg !59
  store i32 7, i32* %localG, align 4, !dbg !59, !tbaa !42
  %15 = load i32, i32* %localG, align 4, !dbg !60, !tbaa !42
  store i32 %15, i32* %localA, align 4, !dbg !60, !tbaa !42
  %16 = bitcast i32* %localG to i8*, !dbg !61
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %16) #4, !dbg !61
  %17 = bitcast i32* %localF to i8*, !dbg !62
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %17) #4, !dbg !62
  %18 = load i32, i32* %localA, align 4, !dbg !63, !tbaa !42
  %cmp1 = icmp eq i32 %18, 7, !dbg !63
  br i1 %cmp1, label %if.then2, label %if.end3, !dbg !63

if.then2:                                         ; preds = %if.end
  %19 = bitcast i32* %localH to i8*, !dbg !64
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %19) #4, !dbg !64
  call void @llvm.dbg.declare(metadata i32* %localH, metadata !34, metadata !DIExpression()), !dbg !64
  store i32 8, i32* %localH, align 4, !dbg !64, !tbaa !42
  %20 = load i32, i32* %localH, align 4, !dbg !65, !tbaa !42
  store i32 %20, i32* %localA, align 4, !dbg !65, !tbaa !42
  %21 = bitcast i32* %localH to i8*, !dbg !66
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %21) #4, !dbg !66
  br label %if.end3, !dbg !66

if.end3:                                          ; preds = %if.then2, %if.end
  %22 = load i32, i32* %localA, align 4, !dbg !67, !tbaa !42
  %cmp4 = icmp ne i32 %22, 8, !dbg !67
  %23 = zext i1 %cmp4 to i64, !dbg !67
  %cond = select i1 %cmp4, i32 -1, i32 0, !dbg !67
  %24 = bitcast i32* %localA to i8*, !dbg !68
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %24) #4, !dbg !68
  ret i32 %cond, !dbg !67
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.expect.i32(i32, i32) #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "C:/path/to/directory", checksumkind: CSK_MD5, checksum: "169b810b4f895de9a9e19d8d0634af5d")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 7.0.0 (trunk)"}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !16)
!9 = !DIFile(filename: "lexicalblock.cxx", directory: "C:/path/to/directory", checksumkind: CSK_MD5, checksum: "169b810b4f895de9a9e19d8d0634af5d")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 32)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !{!17, !18, !19, !20, !22, !25, !27, !30, !32, !34}
!17 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !9, line: 1, type: !13)
!18 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !9, line: 1, type: !12)
!19 = !DILocalVariable(name: "localA", scope: !8, file: !9, line: 2, type: !12)
!20 = !DILocalVariable(name: "localB", scope: !21, file: !9, line: 5, type: !12)
!21 = distinct !DILexicalBlock(scope: !8, file: !9, line: 4)
!22 = !DILocalVariable(name: "localC", scope: !23, file: !9, line: 8, type: !12)
!23 = distinct !DILexicalBlock(scope: !24, file: !9, line: 7)
!24 = distinct !DILexicalBlock(scope: !21, file: !9, line: 7)
!25 = !DILocalVariable(name: "localD", scope: !26, file: !9, line: 13, type: !12)
!26 = distinct !DILexicalBlock(scope: !8, file: !9, line: 12)
!27 = !DILocalVariable(name: "localE", scope: !28, file: !9, line: 19, type: !12)
!28 = distinct !DILexicalBlock(scope: !29, file: !9, line: 18)
!29 = distinct !DILexicalBlock(scope: !8, file: !9, line: 17)
!30 = !DILocalVariable(name: "localF", scope: !31, file: !9, line: 25, type: !12)
!31 = distinct !DILexicalBlock(scope: !8, file: !9, line: 24)
!32 = !DILocalVariable(name: "localG", scope: !33, file: !9, line: 29, type: !12)
!33 = distinct !DILexicalBlock(scope: !31, file: !9, line: 28)
!34 = !DILocalVariable(name: "localH", scope: !35, file: !9, line: 35, type: !12)
!35 = distinct !DILexicalBlock(scope: !36, file: !9, line: 34)
!36 = distinct !DILexicalBlock(scope: !8, file: !9, line: 34)
!37 = !{!38, !38, i64 0}
!38 = !{!"any pointer", !39, i64 0}
!39 = !{!"omnipotent char", !40, i64 0}
!40 = !{!"Simple C++ TBAA"}
!41 = !DILocation(line: 1, scope: !8)
!42 = !{!43, !43, i64 0}
!43 = !{!"int", !39, i64 0}
!44 = !DILocation(line: 2, scope: !8)
!45 = !DILocation(line: 5, scope: !21)
!46 = !DILocation(line: 7, scope: !21)
!47 = !{!"branch_weights", i32 1, i32 2000}
!48 = !DILocation(line: 8, scope: !23)
!49 = !DILocation(line: 9, scope: !23)
!50 = !DILocation(line: 10, scope: !21)
!51 = !DILocation(line: 13, scope: !26)
!52 = !DILocation(line: 14, scope: !26)
!53 = !DILocation(line: 15, scope: !26)
!54 = !DILocation(line: 19, scope: !28)
!55 = !DILocation(line: 20, scope: !28)
!56 = !DILocation(line: 21, scope: !28)
!57 = !DILocation(line: 25, scope: !31)
!58 = !DILocation(line: 26, scope: !31)
!59 = !DILocation(line: 29, scope: !33)
!60 = !DILocation(line: 30, scope: !33)
!61 = !DILocation(line: 31, scope: !33)
!62 = !DILocation(line: 32, scope: !31)
!63 = !DILocation(line: 34, scope: !8)
!64 = !DILocation(line: 35, scope: !35)
!65 = !DILocation(line: 36, scope: !35)
!66 = !DILocation(line: 37, scope: !35)
!67 = !DILocation(line: 39, scope: !8)
!68 = !DILocation(line: 40, scope: !8)
