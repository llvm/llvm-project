; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Verify that we have correct debug info for local variables in code
; instrumented with AddressSanitizer.

; Generated from the source file test.cc:
; int bar(int y) {
;   return y + 2;
; }
; with "clang++ -S -emit-llvm -mllvm -asan-skip-promotable-allocas=0 -fsanitize=address -O0 -g test.cc"

; The address of the (potentially now malloc'ed) alloca ends up
; in RDI, after which it is spilled to the stack. We record the
; spill OFFSET on the stack for checking the debug info below.
; CHECK: #DEBUG_VALUE: bar:y <- [%RDI+0]
; CHECK: movq %rdi, [[OFFSET:[0-9]+]](%rsp)
; CHECK-NEXT: [[START_LABEL:.Ltmp[0-9]+]]
; CHECK-NEXT: #DEBUG_VALUE: bar:y <- [complex expression]
; This location should be valid until the end of the function.

; CHECK:        movq    %rbp, %rsp
; CHECK-NEXT: [[END_LABEL:.Ltmp[0-9]+]]:

; CHECK: .Ldebug_loc{{[0-9]+}}:
; We expect two location ranges for the variable.

; First, its address is stored in %rdi:
; CHECK:      .quad .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT: .quad [[START_LABEL]]-.Lfunc_begin0
; CHECK: DW_OP_breg5

; Then it's addressed via %rsp:
; CHECK:      .quad [[START_LABEL]]-.Lfunc_begin0
; CHECK-NEXT: .quad [[END_LABEL]]-.Lfunc_begin0
; CHECK: DW_OP_breg7
; CHECK-NEXT: [[OFFSET]]
; CHECK: DW_OP_deref

; ModuleID = 'test.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 1, void ()* @asan.module_ctor }]
@__asan_option_detect_stack_use_after_return = external global i32
@__asan_gen_ = private unnamed_addr constant [16 x i8] c"1 32 4 6 y.addr\00", align 1

; Function Attrs: nounwind sanitize_address uwtable
define i32 @_Z3bari(i32 %y) #0 !dbg !4 {
entry:
  %MyAlloca = alloca [64 x i8], align 32
  %0 = ptrtoint [64 x i8]* %MyAlloca to i64
  %1 = load i32, i32* @__asan_option_detect_stack_use_after_return
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %5

; <label>:3                                       ; preds = %entry
  %4 = call i64 @__asan_stack_malloc_0(i64 64, i64 %0)
  br label %5

; <label>:5                                       ; preds = %entry, %3
  %6 = phi i64 [ %0, %entry ], [ %4, %3 ]
  %7 = add i64 %6, 32
  %8 = inttoptr i64 %7 to i32*
  %9 = inttoptr i64 %6 to i64*
  store i64 1102416563, i64* %9
  %10 = add i64 %6, 8
  %11 = inttoptr i64 %10 to i64*
  store i64 ptrtoint ([16 x i8]* @__asan_gen_ to i64), i64* %11
  %12 = add i64 %6, 16
  %13 = inttoptr i64 %12 to i64*
  store i64 ptrtoint (i32 (i32)* @_Z3bari to i64), i64* %13
  %14 = lshr i64 %6, 3
  %15 = add i64 %14, 2147450880
  %16 = add i64 %15, 0
  %17 = inttoptr i64 %16 to i64*
  store i64 -868083100587789839, i64* %17
  %18 = ptrtoint i32* %8 to i64
  %19 = lshr i64 %18, 3
  %20 = add i64 %19, 2147450880
  %21 = inttoptr i64 %20 to i8*
  %22 = load i8, i8* %21
  %23 = icmp ne i8 %22, 0
  call void @llvm.dbg.declare(metadata i32* %8, metadata !12, metadata !14), !dbg !DILocation(scope: !4)
  br i1 %23, label %24, label %30

; <label>:24                                      ; preds = %5
  %25 = and i64 %18, 7
  %26 = add i64 %25, 3
  %27 = trunc i64 %26 to i8
  %28 = icmp sge i8 %27, %22
  br i1 %28, label %29, label %30

; <label>:29                                      ; preds = %24
  call void @__asan_report_store4(i64 %18)
  call void asm sideeffect "", ""()
  unreachable

; <label>:30                                      ; preds = %24, %5
  store i32 %y, i32* %8, align 4
  %31 = ptrtoint i32* %8 to i64, !dbg !13
  %32 = lshr i64 %31, 3, !dbg !13
  %33 = add i64 %32, 2147450880, !dbg !13
  %34 = inttoptr i64 %33 to i8*, !dbg !13
  %35 = load i8, i8* %34, !dbg !13
  %36 = icmp ne i8 %35, 0, !dbg !13
  br i1 %36, label %37, label %43, !dbg !13

; <label>:37                                      ; preds = %30
  %38 = and i64 %31, 7, !dbg !13
  %39 = add i64 %38, 3, !dbg !13
  %40 = trunc i64 %39 to i8, !dbg !13
  %41 = icmp sge i8 %40, %35, !dbg !13
  br i1 %41, label %42, label %43

; <label>:42                                      ; preds = %37
  call void @__asan_report_load4(i64 %31), !dbg !13
  call void asm sideeffect "", ""()
  unreachable

; <label>:43                                      ; preds = %37, %30
  %44 = load i32, i32* %8, align 4, !dbg !13
  %add = add nsw i32 %44, 2, !dbg !13
  store i64 1172321806, i64* %9, !dbg !13
  %45 = icmp ne i64 %6, %0, !dbg !13
  br i1 %45, label %46, label %53, !dbg !13

; <label>:46                                      ; preds = %43
  %47 = add i64 %15, 0, !dbg !13
  %48 = inttoptr i64 %47 to i64*, !dbg !13
  store i64 -723401728380766731, i64* %48, !dbg !13
  %49 = add i64 %6, 56, !dbg !13
  %50 = inttoptr i64 %49 to i64*, !dbg !13
  %51 = load i64, i64* %50, !dbg !13
  %52 = inttoptr i64 %51 to i8*, !dbg !13
  store i8 0, i8* %52, !dbg !13
  br label %56, !dbg !13

; <label>:53                                      ; preds = %43
  %54 = add i64 %15, 0, !dbg !13
  %55 = inttoptr i64 %54 to i64*, !dbg !13
  store i64 0, i64* %55, !dbg !13
  br label %56, !dbg !13

; <label>:56                                      ; preds = %53, %46
  ret i32 %add, !dbg !13
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @asan.module_ctor() {
  call void @__asan_init_v3()
  ret void
}

declare void @__asan_init_v3()

declare void @__asan_report_load4(i64)

declare void @__asan_report_store4(i64)

declare i64 @__asan_stack_malloc_0(i64, i64)

attributes #0 = { nounwind sanitize_address uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (209308)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.cc", directory: "/llvm_cmake_gcc")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bari", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "test.cc", directory: "/llvm_cmake_gcc")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5.0 (209308)"}
!12 = !DILocalVariable(name: "y", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!13 = !DILocation(line: 2, scope: !4)
!14 = !DIExpression(DW_OP_deref)
