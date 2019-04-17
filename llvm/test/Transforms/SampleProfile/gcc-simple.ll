; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/gcc-simple.afdo -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/gcc-simple.afdo -S | FileCheck %s
; XFAIL: powerpc-, powerpc64-, s390x, mips-, mips64-, sparc
; Original code:
;
; #include <stdlib.h>
;
; long long int foo(long i) {
;   if (rand() < 500) return 2; else if (rand() > 5000) return 10; else return 90;
; }
;
; int main() {
;   long long int sum = 0;
;   for (int k = 0; k < 3000; k++)
;     for (int i = 0; i < 200000; i++) sum += foo(i);
;   return sum > 0 ? 0 : 1;
; }
;
; This test was compiled down to bytecode at -O0 to avoid inlining foo() into
; main(). The profile was generated using a GCC-generated binary (also compiled
; at -O0). The conversion from the Linux Perf profile to the GCC autofdo
; profile used the converter at https://github.com/google/autofdo
;
; $ gcc -g -O0 gcc-simple.cc -o gcc-simple
; $ perf record -b ./gcc-simple
; $ create_gcov --binary=gcc-simple --gcov=gcc-simple.afdo

define i64 @_Z3fool(i64 %i) #0 !dbg !4 {
; CHECK: !prof ![[EC1:[0-9]+]]
entry:
  %retval = alloca i64, align 8
  %i.addr = alloca i64, align 8
  store i64 %i, i64* %i.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %i.addr, metadata !16, metadata !17), !dbg !18
  %call = call i32 @rand() #3, !dbg !19
  %cmp = icmp slt i32 %call, 500, !dbg !21
  br i1 %cmp, label %if.then, label %if.else, !dbg !22
; CHECK: !prof ![[PROF1:[0-9]+]]

if.then:                                          ; preds = %entry
  store i64 2, i64* %retval, align 8, !dbg !23
  br label %return, !dbg !23

if.else:                                          ; preds = %entry
  %call1 = call i32 @rand() #3, !dbg !25
  %cmp2 = icmp sgt i32 %call1, 5000, !dbg !28
  br i1 %cmp2, label %if.then.3, label %if.else.4, !dbg !29
; CHECK: !prof ![[PROF2:[0-9]+]]

if.then.3:                                        ; preds = %if.else
  store i64 10, i64* %retval, align 8, !dbg !30
  br label %return, !dbg !30

if.else.4:                                        ; preds = %if.else
  store i64 90, i64* %retval, align 8, !dbg !32
  br label %return, !dbg !32

return:                                           ; preds = %if.else.4, %if.then.3, %if.then
  %0 = load i64, i64* %retval, align 8, !dbg !34
  ret i64 %0, !dbg !34
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i32 @rand() #2

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !9 {
; CHECK: !prof ![[EC2:[0-9]+]]
entry:
  %retval = alloca i32, align 4
  %sum = alloca i64, align 8
  %k = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i64* %sum, metadata !35, metadata !17), !dbg !36
  store i64 0, i64* %sum, align 8, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %k, metadata !37, metadata !17), !dbg !39
  store i32 0, i32* %k, align 4, !dbg !39
  br label %for.cond, !dbg !40

for.cond:                                         ; preds = %for.inc.4, %entry
  %0 = load i32, i32* %k, align 4, !dbg !41
  %cmp = icmp slt i32 %0, 3000, !dbg !45
  br i1 %cmp, label %for.body, label %for.end.6, !dbg !46
; CHECK: !prof ![[PROF3:[0-9]+]]

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.declare(metadata i32* %i, metadata !47, metadata !17), !dbg !49
  store i32 0, i32* %i, align 4, !dbg !49
  br label %for.cond.1, !dbg !50

for.cond.1:                                       ; preds = %for.inc, %for.body
  %1 = load i32, i32* %i, align 4, !dbg !51
  %cmp2 = icmp slt i32 %1, 200000, !dbg !55
  br i1 %cmp2, label %for.body.3, label %for.end, !dbg !56
; CHECK: !prof ![[PROF4:[0-9]+]]

for.body.3:                                       ; preds = %for.cond.1
  %2 = load i32, i32* %i, align 4, !dbg !57
  %conv = sext i32 %2 to i64, !dbg !57
  %call = call i64 @_Z3fool(i64 %conv), !dbg !59
  %3 = load i64, i64* %sum, align 8, !dbg !60
  %add = add nsw i64 %3, %call, !dbg !60
  store i64 %add, i64* %sum, align 8, !dbg !60
  br label %for.inc, !dbg !61

for.inc:                                          ; preds = %for.body.3
  %4 = load i32, i32* %i, align 4, !dbg !62
  %inc = add nsw i32 %4, 1, !dbg !62
  store i32 %inc, i32* %i, align 4, !dbg !62
  br label %for.cond.1, !dbg !64

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.4, !dbg !65

for.inc.4:                                        ; preds = %for.end
  %5 = load i32, i32* %k, align 4, !dbg !67
  %inc5 = add nsw i32 %5, 1, !dbg !67
  store i32 %inc5, i32* %k, align 4, !dbg !67
  br label %for.cond, !dbg !68

for.end.6:                                        ; preds = %for.cond
  %6 = load i64, i64* %sum, align 8, !dbg !69
  %cmp7 = icmp sgt i64 %6, 0, !dbg !70
  %cond = select i1 %cmp7, i32 0, i32 1, !dbg !69
  ret i32 %cond, !dbg !71
}

; CHECK ![[EC1]] = !{!"function_entry_count", i64 24108}
; CHECK ![[PROF1]] = !{!"branch_weights", i32 1, i32 30124}
; CHECK ![[PROF2]] = !{!"branch_weights", i32 30177, i32 29579}
; CHECK ![[EC2]] = !{!"function_entry_count", i64 0}
; CHECK ![[PROF3]] = !{!"branch_weights", i32 1, i32 1}
; CHECK ![[PROF4]] = !{!"branch_weights", i32 1, i32 20238}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 247554) (llvm/trunk 247557)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "discriminator.cc", directory: "/usr/local/google/home/dnovillo/llvm/test/autofdo")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8}
!7 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !10, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.8.0 (trunk 247554) (llvm/trunk 247557)"}
!16 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 3, type: !8)
!17 = !DIExpression()
!18 = !DILocation(line: 3, column: 24, scope: !4)
!19 = !DILocation(line: 4, column: 7, scope: !20)
!20 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 7)
!21 = !DILocation(line: 4, column: 14, scope: !20)
!22 = !DILocation(line: 4, column: 7, scope: !4)
!23 = !DILocation(line: 4, column: 21, scope: !24)
!24 = !DILexicalBlockFile(scope: !20, file: !1, discriminator: 1)
!25 = !DILocation(line: 4, column: 40, scope: !26)
!26 = !DILexicalBlockFile(scope: !27, file: !1, discriminator: 2)
!27 = distinct !DILexicalBlock(scope: !20, file: !1, line: 4, column: 40)
!28 = !DILocation(line: 4, column: 47, scope: !27)
!29 = !DILocation(line: 4, column: 40, scope: !20)
!30 = !DILocation(line: 4, column: 55, scope: !31)
!31 = !DILexicalBlockFile(scope: !27, file: !1, discriminator: 3)
!32 = !DILocation(line: 4, column: 71, scope: !33)
!33 = !DILexicalBlockFile(scope: !27, file: !1, discriminator: 4)
!34 = !DILocation(line: 5, column: 1, scope: !4)
!35 = !DILocalVariable(name: "sum", scope: !9, file: !1, line: 8, type: !7)
!36 = !DILocation(line: 8, column: 17, scope: !9)
!37 = !DILocalVariable(name: "k", scope: !38, file: !1, line: 9, type: !12)
!38 = distinct !DILexicalBlock(scope: !9, file: !1, line: 9, column: 3)
!39 = !DILocation(line: 9, column: 12, scope: !38)
!40 = !DILocation(line: 9, column: 8, scope: !38)
!41 = !DILocation(line: 9, column: 19, scope: !42)
!42 = !DILexicalBlockFile(scope: !43, file: !1, discriminator: 2)
!43 = !DILexicalBlockFile(scope: !44, file: !1, discriminator: 1)
!44 = distinct !DILexicalBlock(scope: !38, file: !1, line: 9, column: 3)
!45 = !DILocation(line: 9, column: 21, scope: !44)
!46 = !DILocation(line: 9, column: 3, scope: !38)
!47 = !DILocalVariable(name: "i", scope: !48, file: !1, line: 10, type: !12)
!48 = distinct !DILexicalBlock(scope: !44, file: !1, line: 10, column: 5)
!49 = !DILocation(line: 10, column: 14, scope: !48)
!50 = !DILocation(line: 10, column: 10, scope: !48)
!51 = !DILocation(line: 10, column: 21, scope: !52)
!52 = !DILexicalBlockFile(scope: !53, file: !1, discriminator: 5)
!53 = !DILexicalBlockFile(scope: !54, file: !1, discriminator: 1)
!54 = distinct !DILexicalBlock(scope: !48, file: !1, line: 10, column: 5)
!55 = !DILocation(line: 10, column: 23, scope: !54)
!56 = !DILocation(line: 10, column: 5, scope: !48)
!57 = !DILocation(line: 10, column: 49, scope: !58)
!58 = !DILexicalBlockFile(scope: !54, file: !1, discriminator: 2)
!59 = !DILocation(line: 10, column: 45, scope: !54)
!60 = !DILocation(line: 10, column: 42, scope: !54)
!61 = !DILocation(line: 10, column: 38, scope: !54)
!62 = !DILocation(line: 10, column: 34, scope: !63)
!63 = !DILexicalBlockFile(scope: !54, file: !1, discriminator: 4)
!64 = !DILocation(line: 10, column: 5, scope: !54)
!65 = !DILocation(line: 10, column: 50, scope: !66)
!66 = !DILexicalBlockFile(scope: !48, file: !1, discriminator: 3)
!67 = !DILocation(line: 9, column: 30, scope: !44)
!68 = !DILocation(line: 9, column: 3, scope: !44)
!69 = !DILocation(line: 11, column: 10, scope: !9)
!70 = !DILocation(line: 11, column: 14, scope: !9)
!71 = !DILocation(line: 11, column: 3, scope: !9)
