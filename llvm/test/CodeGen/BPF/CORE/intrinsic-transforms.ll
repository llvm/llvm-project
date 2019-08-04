; RUN: opt -O2 -march=bpfeb < %s | llvm-dis | FileCheck %s
; RUN: opt -O2 -march=bpfel < %s | llvm-dis | FileCheck %s
;
; Source code:
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const int *arg);
;   int test(int b, int *arg) {
;     int v1 = b ? get_value(_(&arg[4])) : 0;
;     int v2 = b ? get_value(_(&arg[4])) : 0;
;     return v1 + v2;
;   }
; Compilation flag:
;   clang -target bpf -O0 -g -S -emit-llvm -Xclang -disable-O0-optnone test.c

; Function Attrs: noinline nounwind
define dso_local i32 @test(i32 %b, i32* %arg) #0 !dbg !10 {
entry:
  %b.addr = alloca i32, align 4
  %arg.addr = alloca i32*, align 8
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !13, metadata !DIExpression()), !dbg !14
  store i32* %arg, i32** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arg.addr, metadata !15, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32* %v1, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = load i32, i32* %b.addr, align 4, !dbg !19
  %tobool = icmp ne i32 %0, 0, !dbg !19
  br i1 %tobool, label %cond.true, label %cond.false, !dbg !19

cond.true:                                        ; preds = %entry
  %1 = load i32*, i32** %arg.addr, align 8, !dbg !20
  %2 = call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* %1, i32 0, i32 4), !dbg !20, !llvm.preserve.access.index !4
  %3 = bitcast i32* %2 to i8*, !dbg !20
  %4 = bitcast i8* %3 to i32*, !dbg !20
  %call = call i32 @get_value(i32* %4), !dbg !21
  br label %cond.end, !dbg !19

cond.false:                                       ; preds = %entry
  br label %cond.end, !dbg !19

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %call, %cond.true ], [ 0, %cond.false ], !dbg !19
  store i32 %cond, i32* %v1, align 4, !dbg !18
  call void @llvm.dbg.declare(metadata i32* %v2, metadata !22, metadata !DIExpression()), !dbg !23
  %5 = load i32, i32* %b.addr, align 4, !dbg !24
  %tobool1 = icmp ne i32 %5, 0, !dbg !24
  br i1 %tobool1, label %cond.true2, label %cond.false4, !dbg !24

cond.true2:                                       ; preds = %cond.end
  %6 = load i32*, i32** %arg.addr, align 8, !dbg !25
  %7 = call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* %6, i32 0, i32 4), !dbg !25, !llvm.preserve.access.index !4
  %8 = bitcast i32* %7 to i8*, !dbg !25
  %9 = bitcast i8* %8 to i32*, !dbg !25
  %call3 = call i32 @get_value(i32* %9), !dbg !26
  br label %cond.end5, !dbg !24

; CHECK: tail call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* %{{[0-9a-z]+}}, i32 0, i32 4), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
; CHECK-NOT: tail call i32* @llvm.preserve.array.access.index

cond.false4:                                      ; preds = %cond.end
  br label %cond.end5, !dbg !24

cond.end5:                                        ; preds = %cond.false4, %cond.true2
  %cond6 = phi i32 [ %call3, %cond.true2 ], [ 0, %cond.false4 ], !dbg !24
  store i32 %cond6, i32* %v2, align 4, !dbg !23
  %10 = load i32, i32* %v1, align 4, !dbg !27
  %11 = load i32, i32* %v2, align 4, !dbg !28
  %add = add nsw i32 %10, %11, !dbg !29
  ret i32 %add, !dbg !30
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @get_value(i32*) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32*, i32 immarg, i32 immarg) #3

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4618b07fe2cede1b73512d1c260cf4981661f47f)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4618b07fe2cede1b73512d1c260cf4981661f47f)"}
!10 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, isDefinition: true, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!5, !5, !4}
!13 = !DILocalVariable(name: "b", arg: 1, scope: !10, file: !1, line: 3, type: !5)
!14 = !DILocation(line: 3, column: 14, scope: !10)
!15 = !DILocalVariable(name: "arg", arg: 2, scope: !10, file: !1, line: 3, type: !4)
!16 = !DILocation(line: 3, column: 22, scope: !10)
!17 = !DILocalVariable(name: "v1", scope: !10, file: !1, line: 4, type: !5)
!18 = !DILocation(line: 4, column: 7, scope: !10)
!19 = !DILocation(line: 4, column: 12, scope: !10)
!20 = !DILocation(line: 4, column: 26, scope: !10)
!21 = !DILocation(line: 4, column: 16, scope: !10)
!22 = !DILocalVariable(name: "v2", scope: !10, file: !1, line: 5, type: !5)
!23 = !DILocation(line: 5, column: 7, scope: !10)
!24 = !DILocation(line: 5, column: 12, scope: !10)
!25 = !DILocation(line: 5, column: 26, scope: !10)
!26 = !DILocation(line: 5, column: 16, scope: !10)
!27 = !DILocation(line: 6, column: 10, scope: !10)
!28 = !DILocation(line: 6, column: 15, scope: !10)
!29 = !DILocation(line: 6, column: 13, scope: !10)
!30 = !DILocation(line: 6, column: 3, scope: !10)
