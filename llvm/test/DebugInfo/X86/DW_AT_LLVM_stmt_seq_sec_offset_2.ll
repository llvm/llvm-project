; RUN: llc -O3 -mtriple=arm64-apple-macosx11.0.0 -o %t_yes -filetype=obj %s -emit-func-debug-line-table-offsets
; RUN: llvm-dwarfdump -verify %t_yes

; generated from:
; #define ATTRIB extern "C" __attribute__((noinline))
; volatile int global_result = 0;
;
; ATTRIB int function1_copy1(int a) {
;   return ++a;
; }
;
; ATTRIB int function3_copy1(int a) {
;     int b = a + 3;
;     return b + 1;
; }
;
; ATTRIB int function2_copy1(int a) {
;     return a - 22;
; }
;
; ATTRIB int function3_copy2(int a) {
;     int b = a + 3;
;     return b + 1;
; }
;
; ATTRIB int function2_copy2(int a) {
;     int result = a - 22;
;     return result;
; }
;
; struct logic_error {
;     logic_error(const char* s) {}
; };
;
; struct length_error : public logic_error {
;     __attribute__((noinline)) explicit length_error(const char* s) : logic_error(s) {}
; };
;
; int main() {
;     int sum = 0;
;     sum += function2_copy2(3);
;     sum += function3_copy2(41);
;     sum += function2_copy1(11);
;     sum += function1_copy1(42);
;     length_error e("test");
;     return sum;
; }
; =====================

; ModuleID = 'stmt-seq-macho.cpp'
source_filename = "stmt-seq-macho.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"

%struct.length_error = type { i8 }

@.str = private unnamed_addr constant [5 x i8] c"test\00", align 1, !dbg !0

; Function Attrs: minsize mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none)
define range(i32 -2147483647, -2147483648) i32 @function1_copy1(i32 noundef %a) local_unnamed_addr #0 !dbg !17 {
entry:
    #dbg_value(i32 %a, !22, !DIExpression(), !23)
  %inc = add nsw i32 %a, 1, !dbg !24
    #dbg_value(i32 %inc, !22, !DIExpression(), !23)
  ret i32 %inc, !dbg !25
}

; Function Attrs: minsize mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none)
define range(i32 -2147483644, -2147483648) i32 @function3_copy1(i32 noundef %a) local_unnamed_addr #0 !dbg !26 {
entry:
    #dbg_value(i32 %a, !28, !DIExpression(), !30)
    #dbg_value(i32 %a, !29, !DIExpression(DW_OP_plus_uconst, 3, DW_OP_stack_value), !30)
  %add1 = add nsw i32 %a, 4, !dbg !31
  ret i32 %add1, !dbg !32
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: minsize mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none)
define range(i32 -2147483648, 2147483626) i32 @function2_copy1(i32 noundef %a) local_unnamed_addr #0 !dbg !33 {
entry:
    #dbg_value(i32 %a, !35, !DIExpression(), !36)
  %sub = add nsw i32 %a, -22, !dbg !37
  ret i32 %sub, !dbg !38
}

; Function Attrs: minsize mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none)
define range(i32 -2147483644, -2147483648) i32 @function3_copy2(i32 noundef %a) local_unnamed_addr #0 !dbg !39 {
entry:
    #dbg_value(i32 %a, !41, !DIExpression(), !43)
    #dbg_value(i32 %a, !42, !DIExpression(DW_OP_plus_uconst, 3, DW_OP_stack_value), !43)
  %add1 = add nsw i32 %a, 4, !dbg !44
  ret i32 %add1, !dbg !45
}

; Function Attrs: minsize mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none)
define range(i32 -2147483648, 2147483626) i32 @function2_copy2(i32 noundef %a) local_unnamed_addr #0 !dbg !46 {
entry:
    #dbg_value(i32 %a, !48, !DIExpression(), !50)
  %sub = add nsw i32 %a, -22, !dbg !51
    #dbg_value(i32 %sub, !49, !DIExpression(), !50)
  ret i32 %sub, !dbg !52
}

; Function Attrs: minsize mustprogress norecurse nounwind optsize ssp
define noundef i32 @main() local_unnamed_addr #2 !dbg !53 {
entry:
  %e = alloca %struct.length_error, align 1
    #dbg_value(i32 0, !57, !DIExpression(), !73)
  %call = tail call i32 @function2_copy2(i32 noundef 3) #4, !dbg !74
    #dbg_value(i32 %call, !57, !DIExpression(), !73)
  %call1 = tail call i32 @function3_copy2(i32 noundef 41) #4, !dbg !75
  %add2 = add nsw i32 %call1, %call, !dbg !76
    #dbg_value(i32 %add2, !57, !DIExpression(), !73)
  %call3 = tail call i32 @function2_copy1(i32 noundef 11) #4, !dbg !77
  %add4 = add nsw i32 %add2, %call3, !dbg !78
    #dbg_value(i32 %add4, !57, !DIExpression(), !73)
  %call5 = tail call i32 @function1_copy1(i32 noundef 42) #4, !dbg !79
  %add6 = add nsw i32 %add4, %call5, !dbg !80
    #dbg_value(i32 %add6, !57, !DIExpression(), !73)
  call void @llvm.lifetime.start.p0(ptr nonnull %e) #5, !dbg !81
    #dbg_declare(ptr %e, !58, !DIExpression(), !82)
  %call7 = call noundef ptr @_ZN12length_errorC1EPKc(ptr noundef nonnull align 1 dereferenceable(1) %e, ptr noundef nonnull @.str) #4, !dbg !82
  call void @llvm.lifetime.end.p0(ptr nonnull %e) #5, !dbg !83
  ret i32 %add6, !dbg !84
}

; Function Attrs: minsize mustprogress noinline nounwind optsize ssp
define linkonce_odr noundef ptr @_ZN12length_errorC1EPKc(ptr noundef nonnull returned align 1 dereferenceable(1) %this, ptr noundef %s) unnamed_addr #3 !dbg !85 {
entry:
    #dbg_value(ptr %this, !87, !DIExpression(), !90)
    #dbg_value(ptr %s, !89, !DIExpression(), !90)
  %call = tail call noundef ptr @_ZN12length_errorC2EPKc(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %s) #4, !dbg !91
  ret ptr %this, !dbg !92
}

; Function Attrs: minsize mustprogress noinline nounwind optsize ssp
define linkonce_odr noundef ptr @_ZN12length_errorC2EPKc(ptr noundef nonnull returned align 1 dereferenceable(1) %this, ptr noundef %s) unnamed_addr #3 !dbg !93 {
entry:
    #dbg_value(ptr %this, !95, !DIExpression(), !97)
    #dbg_value(ptr %s, !96, !DIExpression(), !97)
  ret ptr %this, !dbg !98
}

attributes #0 = { minsize mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { minsize mustprogress norecurse nounwind optsize ssp "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" }
attributes #3 = { minsize mustprogress noinline nounwind optsize ssp "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" }
attributes #4 = { minsize optsize }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, line: 51, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "stmt-seq-macho.cpp", directory: "")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 40, elements: !6)
!4 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{!7}
!7 = !DISubrange(count: 5)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !9, producer: "clang version 22.0.0git (git@github.com:DataCorrupted/llvm-project.git cedce2128dc872a2f1024c9907fd78bdee4b7fe7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !10, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!9 = !DIFile(filename: "stmt-seq-macho.cpp", directory: "/private/tmp/stmt_seq")
!10 = !{!0}
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 7, !"frame-pointer", i32 1}
!16 = !{!"clang version 22.0.0git (git@github.com:DataCorrupted/llvm-project.git cedce2128dc872a2f1024c9907fd78bdee4b7fe7)"}
!17 = distinct !DISubprogram(name: "function1_copy1", scope: !2, file: !2, line: 14, type: !18, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !21)
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !{!22}
!22 = !DILocalVariable(name: "a", arg: 1, scope: !17, file: !2, line: 14, type: !20)
!23 = !DILocation(line: 0, scope: !17)
!24 = !DILocation(line: 15, column: 10, scope: !17)
!25 = !DILocation(line: 15, column: 3, scope: !17)
!26 = distinct !DISubprogram(name: "function3_copy1", scope: !2, file: !2, line: 18, type: !18, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !27)
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "a", arg: 1, scope: !26, file: !2, line: 18, type: !20)
!29 = !DILocalVariable(name: "b", scope: !26, file: !2, line: 19, type: !20)
!30 = !DILocation(line: 0, scope: !26)
!31 = !DILocation(line: 20, column: 14, scope: !26)
!32 = !DILocation(line: 20, column: 5, scope: !26)
!33 = distinct !DISubprogram(name: "function2_copy1", scope: !2, file: !2, line: 23, type: !18, scopeLine: 23, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !34)
!34 = !{!35}
!35 = !DILocalVariable(name: "a", arg: 1, scope: !33, file: !2, line: 23, type: !20)
!36 = !DILocation(line: 0, scope: !33)
!37 = !DILocation(line: 24, column: 14, scope: !33)
!38 = !DILocation(line: 24, column: 5, scope: !33)
!39 = distinct !DISubprogram(name: "function3_copy2", scope: !2, file: !2, line: 27, type: !18, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !40)
!40 = !{!41, !42}
!41 = !DILocalVariable(name: "a", arg: 1, scope: !39, file: !2, line: 27, type: !20)
!42 = !DILocalVariable(name: "b", scope: !39, file: !2, line: 28, type: !20)
!43 = !DILocation(line: 0, scope: !39)
!44 = !DILocation(line: 29, column: 14, scope: !39)
!45 = !DILocation(line: 29, column: 5, scope: !39)
!46 = distinct !DISubprogram(name: "function2_copy2", scope: !2, file: !2, line: 32, type: !18, scopeLine: 32, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !47)
!47 = !{!48, !49}
!48 = !DILocalVariable(name: "a", arg: 1, scope: !46, file: !2, line: 32, type: !20)
!49 = !DILocalVariable(name: "result", scope: !46, file: !2, line: 33, type: !20)
!50 = !DILocation(line: 0, scope: !46)
!51 = !DILocation(line: 33, column: 20, scope: !46)
!52 = !DILocation(line: 34, column: 5, scope: !46)
!53 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 45, type: !54, scopeLine: 45, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !56)
!54 = !DISubroutineType(types: !55)
!55 = !{!20}
!56 = !{!57, !58}
!57 = !DILocalVariable(name: "sum", scope: !53, file: !2, line: 46, type: !20)
!58 = !DILocalVariable(name: "e", scope: !53, file: !2, line: 51, type: !59)
!59 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "length_error", file: !2, line: 41, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !60, identifier: "_ZTS12length_error")
!60 = !{!61, !69}
!61 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !59, baseType: !62, extraData: i32 0)
!62 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "logic_error", file: !2, line: 37, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !63, identifier: "_ZTS11logic_error")
!63 = !{!64}
!64 = !DISubprogram(name: "logic_error", scope: !62, file: !2, line: 38, type: !65, scopeLine: 38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!65 = !DISubroutineType(types: !66)
!66 = !{null, !67, !68}
!67 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !62, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!68 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!69 = !DISubprogram(name: "length_error", scope: !59, file: !2, line: 42, type: !70, scopeLine: 42, flags: DIFlagExplicit | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!70 = !DISubroutineType(types: !71)
!71 = !{null, !72, !68}
!72 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !59, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!73 = !DILocation(line: 0, scope: !53)
!74 = !DILocation(line: 47, column: 12, scope: !53)
!75 = !DILocation(line: 48, column: 12, scope: !53)
!76 = !DILocation(line: 48, column: 9, scope: !53)
!77 = !DILocation(line: 49, column: 12, scope: !53)
!78 = !DILocation(line: 49, column: 9, scope: !53)
!79 = !DILocation(line: 50, column: 12, scope: !53)
!80 = !DILocation(line: 50, column: 9, scope: !53)
!81 = !DILocation(line: 51, column: 5, scope: !53)
!82 = !DILocation(line: 51, column: 18, scope: !53)
!83 = !DILocation(line: 53, column: 1, scope: !53)
!84 = !DILocation(line: 52, column: 5, scope: !53)
!85 = distinct !DISubprogram(name: "length_error", linkageName: "_ZN12length_errorC1EPKc", scope: !59, file: !2, line: 42, type: !70, scopeLine: 42, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, declaration: !69, retainedNodes: !86)
!86 = !{!87, !89}
!87 = !DILocalVariable(name: "this", arg: 1, scope: !85, type: !88, flags: DIFlagArtificial | DIFlagObjectPointer)
!88 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !59, size: 64)
!89 = !DILocalVariable(name: "s", arg: 2, scope: !85, file: !2, line: 42, type: !68)
!90 = !DILocation(line: 0, scope: !85)
!91 = !DILocation(line: 42, column: 85, scope: !85)
!92 = !DILocation(line: 42, column: 86, scope: !85)
!93 = distinct !DISubprogram(name: "length_error", linkageName: "_ZN12length_errorC2EPKc", scope: !59, file: !2, line: 42, type: !70, scopeLine: 42, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, declaration: !69, retainedNodes: !94)
!94 = !{!95, !96}
!95 = !DILocalVariable(name: "this", arg: 1, scope: !93, type: !88, flags: DIFlagArtificial | DIFlagObjectPointer)
!96 = !DILocalVariable(name: "s", arg: 2, scope: !93, file: !2, line: 42, type: !68)
!97 = !DILocation(line: 0, scope: !93)
!98 = !DILocation(line: 42, column: 86, scope: !93)
