; RUN: llc -march=x86-64 -mtriple=x86_64-linux < %s | FileCheck %s
; RUN: opt -strip-debug < %s | llc -march=x86-64 -mtriple=x86_64-linux | FileCheck %s
; http://llvm.org/PR19051. Minor code-motion difference with -g.
; Presence of debug info shouldn't affect the codegen. Make sure that
; we generated the same code sequence with and without debug info. 
;
; CHECK:      callq   _Z3fooPcjPKc
; CHECK:      callq   _Z3fooPcjPKc
; CHECK:      movq    %rsp, %rdi
; CHECK:      movl    $4, %esi
; CHECK:      testl   {{%[a-z]+}}, {{%[a-z]+}}
; CHECK:      je     .LBB0_4

; Regenerate test with this command: 
;   clang++ -emit-llvm -S -O2 -g
; from this source:
;
; extern void foo(char *dst,unsigned siz,const char *src);
; extern const char * i2str(int);
;
; struct AAA3 {
;  AAA3(const char *value) { foo(text,sizeof(text),value);}
;  void operator=(const char *value) { foo(text,sizeof(text),value);}
;  operator const char*() const { return text;}
;  char text[4];
; };
;
; void bar (int param1,int param2)  {
;   const char * temp(0);
;
;   if (param2) {
;     temp = i2str(param2);
;   }
;   AAA3 var1("");
;   AAA3 var2("");
;
;   if (param1)
;     var2 = "+";
;   else
;     var2 = "-";
;   var1 = "";
; }

%struct.AAA3 = type { [4 x i8] }

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"-\00", align 1

; Function Attrs: uwtable
define void @_Z3barii(i32 %param1, i32 %param2) #0 !dbg !24 {
entry:
  %var1 = alloca %struct.AAA3, align 1
  %var2 = alloca %struct.AAA3, align 1
  tail call void @llvm.dbg.value(metadata i32 %param1, i64 0, metadata !29, metadata !46), !dbg !47
  tail call void @llvm.dbg.value(metadata i32 %param2, i64 0, metadata !30, metadata !46), !dbg !48
  tail call void @llvm.dbg.value(metadata i8* null, i64 0, metadata !31, metadata !46), !dbg !49
  %tobool = icmp eq i32 %param2, 0, !dbg !50
  br i1 %tobool, label %if.end, label %if.then, !dbg !52

if.then:                                          ; preds = %entry
  %call = tail call i8* @_Z5i2stri(i32 %param2), !dbg !53
  tail call void @llvm.dbg.value(metadata i8* %call, i64 0, metadata !31, metadata !46), !dbg !49
  br label %if.end, !dbg !55

if.end:                                           ; preds = %entry, %if.then
  %0 = getelementptr inbounds %struct.AAA3, %struct.AAA3* %var1, i64 0, i32 0, i64 0, !dbg !56
  call void @llvm.lifetime.start(i64 4, i8* %0) #4, !dbg !56
  tail call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !32, metadata !57), !dbg !58
  tail call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !36, metadata !46), !dbg !59
  tail call void @llvm.dbg.value(metadata i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0), i64 0, metadata !38, metadata !46), !dbg !62
  call void @_Z3fooPcjPKc(i8* %0, i32 4, i8* nonnull getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0)), !dbg !63
  %1 = getelementptr inbounds %struct.AAA3, %struct.AAA3* %var2, i64 0, i32 0, i64 0, !dbg !65
  call void @llvm.lifetime.start(i64 4, i8* %1) #4, !dbg !65
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !33, metadata !57), !dbg !66
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !36, metadata !46), !dbg !67
  call void @llvm.dbg.value(metadata i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0), i64 0, metadata !38, metadata !46), !dbg !69
  call void @_Z3fooPcjPKc(i8* %1, i32 4, i8* nonnull getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0)), !dbg !70
  %tobool1 = icmp eq i32 %param1, 0, !dbg !71
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !33, metadata !57), !dbg !66
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !41, metadata !46), !dbg !73
  br i1 %tobool1, label %if.else, label %if.then2, !dbg !75

if.then2:                                         ; preds = %if.end
  call void @llvm.dbg.value(metadata i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i64 0, metadata !42, metadata !46), !dbg !76
  call void @_Z3fooPcjPKc(i8* %1, i32 4, i8* nonnull getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0)), !dbg !78
  br label %if.end3, !dbg !79

if.else:                                          ; preds = %if.end
  call void @llvm.dbg.value(metadata i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), i64 0, metadata !42, metadata !46), !dbg !80
  call void @_Z3fooPcjPKc(i8* %1, i32 4, i8* nonnull getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0)), !dbg !81
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !32, metadata !57), !dbg !58
  call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !41, metadata !46), !dbg !82
  call void @llvm.dbg.value(metadata i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0), i64 0, metadata !42, metadata !46), !dbg !84
  call void @_Z3fooPcjPKc(i8* %0, i32 4, i8* nonnull getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0)), !dbg !85
  call void @llvm.lifetime.end(i64 4, i8* %1) #4, !dbg !86
  call void @llvm.lifetime.end(i64 4, i8* %0) #4, !dbg !87
  ret void, !dbg !86
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare i8* @_Z5i2stri(i32) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

declare void @_Z3fooPcjPKc(i8*, i32, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #3

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!43, !44}
!llvm.ident = !{!45}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 255993) (llvm/trunk 256074)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.cpp", directory: "/mnt/extra")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "AAA3", file: !1, line: 4, size: 32, align: 8, elements: !5, identifier: "_ZTS4AAA3")
!5 = !{!6, !11, !17, !18}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "text", scope: !4, file: !1, line: 8, baseType: !7, size: 32, align: 8)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 32, align: 8, elements: !9)
!8 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!9 = !{!10}
!10 = !DISubrange(count: 4)
!11 = !DISubprogram(name: "AAA3", scope: !4, file: !1, line: 5, type: !12, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !15}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!17 = !DISubprogram(name: "operator=", linkageName: "_ZN4AAA3aSEPKc", scope: !4, file: !1, line: 6, type: !12, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true)
!18 = !DISubprogram(name: "operator const char *", linkageName: "_ZNK4AAA3cvPKcEv", scope: !4, file: !1, line: 7, type: !19, isLocal: false, isDefinition: false, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true)
!19 = !DISubroutineType(types: !20)
!20 = !{!15, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !4)
!24 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barii", scope: !1, file: !1, line: 11, type: !25, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !28)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !27, !27}
!27 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!28 = !{!29, !30, !31, !32, !33}
!29 = !DILocalVariable(name: "param1", arg: 1, scope: !24, file: !1, line: 11, type: !27)
!30 = !DILocalVariable(name: "param2", arg: 2, scope: !24, file: !1, line: 11, type: !27)
!31 = !DILocalVariable(name: "temp", scope: !24, file: !1, line: 12, type: !15)
!32 = !DILocalVariable(name: "var1", scope: !24, file: !1, line: 17, type: !4)
!33 = !DILocalVariable(name: "var2", scope: !24, file: !1, line: 18, type: !4)
!34 = distinct !DISubprogram(name: "AAA3", linkageName: "_ZN4AAA3C2EPKc", scope: !4, file: !1, line: 5, type: !12, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !11, variables: !35)
!35 = !{!36, !38}
!36 = !DILocalVariable(name: "this", arg: 1, scope: !34, type: !37, flags: DIFlagArtificial | DIFlagObjectPointer)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64)
!38 = !DILocalVariable(name: "value", arg: 2, scope: !34, file: !1, line: 5, type: !15)
!39 = distinct !DISubprogram(name: "operator=", linkageName: "_ZN4AAA3aSEPKc", scope: !4, file: !1, line: 6, type: !12, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !17, variables: !40)
!40 = !{!41, !42}
!41 = !DILocalVariable(name: "this", arg: 1, scope: !39, type: !37, flags: DIFlagArtificial | DIFlagObjectPointer)
!42 = !DILocalVariable(name: "value", arg: 2, scope: !39, file: !1, line: 6, type: !15)
!43 = !{i32 2, !"Dwarf Version", i32 4}
!44 = !{i32 2, !"Debug Info Version", i32 3}
!45 = !{!"clang version 3.8.0 (trunk 255993) (llvm/trunk 256074)"}
!46 = !DIExpression()
!47 = !DILocation(line: 11, column: 15, scope: !24)
!48 = !DILocation(line: 11, column: 26, scope: !24)
!49 = !DILocation(line: 12, column: 16, scope: !24)
!50 = !DILocation(line: 14, column: 7, scope: !51)
!51 = distinct !DILexicalBlock(scope: !24, file: !1, line: 14, column: 7)
!52 = !DILocation(line: 14, column: 7, scope: !24)
!53 = !DILocation(line: 15, column: 12, scope: !54)
!54 = distinct !DILexicalBlock(scope: !51, file: !1, line: 14, column: 15)
!55 = !DILocation(line: 16, column: 3, scope: !54)
!56 = !DILocation(line: 17, column: 3, scope: !24)
!57 = !DIExpression(DW_OP_deref)
!58 = !DILocation(line: 17, column: 8, scope: !24)
!59 = !DILocation(line: 0, scope: !34, inlinedAt: !60)
!60 = distinct !DILocation(line: 17, column: 8, scope: !61)
!61 = !DILexicalBlockFile(scope: !24, file: !1, discriminator: 1)
!62 = !DILocation(line: 5, column: 19, scope: !34, inlinedAt: !60)
!63 = !DILocation(line: 5, column: 28, scope: !64, inlinedAt: !60)
!64 = distinct !DILexicalBlock(scope: !34, file: !1, line: 5, column: 26)
!65 = !DILocation(line: 18, column: 3, scope: !24)
!66 = !DILocation(line: 18, column: 8, scope: !24)
!67 = !DILocation(line: 0, scope: !34, inlinedAt: !68)
!68 = distinct !DILocation(line: 18, column: 8, scope: !61)
!69 = !DILocation(line: 5, column: 19, scope: !34, inlinedAt: !68)
!70 = !DILocation(line: 5, column: 28, scope: !64, inlinedAt: !68)
!71 = !DILocation(line: 20, column: 7, scope: !72)
!72 = distinct !DILexicalBlock(scope: !24, file: !1, line: 20, column: 7)
!73 = !DILocation(line: 0, scope: !39, inlinedAt: !74)
!74 = distinct !DILocation(line: 23, column: 10, scope: !72)
!75 = !DILocation(line: 20, column: 7, scope: !24)
!76 = !DILocation(line: 6, column: 29, scope: !39, inlinedAt: !77)
!77 = distinct !DILocation(line: 21, column: 10, scope: !72)
!78 = !DILocation(line: 6, column: 38, scope: !39, inlinedAt: !77)
!79 = !DILocation(line: 21, column: 5, scope: !72)
!80 = !DILocation(line: 6, column: 29, scope: !39, inlinedAt: !74)
!81 = !DILocation(line: 6, column: 38, scope: !39, inlinedAt: !74)
!82 = !DILocation(line: 0, scope: !39, inlinedAt: !83)
!83 = distinct !DILocation(line: 24, column: 8, scope: !24)
!84 = !DILocation(line: 6, column: 29, scope: !39, inlinedAt: !83)
!85 = !DILocation(line: 6, column: 38, scope: !39, inlinedAt: !83)
!86 = !DILocation(line: 25, column: 1, scope: !24)
!87 = !DILocation(line: 25, column: 1, scope: !61)
