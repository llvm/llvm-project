; RUN: llc -filetype=obj -O0 < %s
; Test that we handle DBG_VALUEs in a register without crashing.
;
; Generated from clang with -fsanitize=address:
; struct A {
;   A();
;   A(const A&);
; };
;
; A func(int) {
;   A a;
;   return a;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i8 }

@__asan_mapping_offset = linkonce_odr constant i64 2147450880
@__asan_mapping_scale = linkonce_odr constant i64 3
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 1, void ()* @asan.module_ctor }]
@__asan_gen_ = private unnamed_addr constant [16 x i8] c"1 32 4 5 .addr \00", align 1

; Function Attrs: sanitize_address uwtable
define void @_Z4funci(%struct.A* noalias sret %agg.result, i32) #0 "stack-protector-buffer-size"="1" !dbg !4 {
entry:
  %MyAlloca = alloca [96 x i8], align 32
  %1 = ptrtoint [96 x i8]* %MyAlloca to i64
  %2 = add i64 %1, 32
  %3 = inttoptr i64 %2 to i32*
  %4 = inttoptr i64 %1 to i64*
  store i64 1102416563, i64* %4
  %5 = add i64 %1, 8
  %6 = inttoptr i64 %5 to i64*
  store i64 ptrtoint ([16 x i8]* @__asan_gen_ to i64), i64* %6
  %7 = add i64 %1, 16
  %8 = inttoptr i64 %7 to i64*
  store i64 ptrtoint (void (%struct.A*, i32)* @_Z4funci to i64), i64* %8
  %9 = lshr i64 %1, 3
  %10 = add i64 %9, 2147450880
  %11 = inttoptr i64 %10 to i32*
  store i32 -235802127, i32* %11
  %12 = add i64 %10, 4
  %13 = inttoptr i64 %12 to i32*
  store i32 -185273340, i32* %13
  %14 = add i64 %10, 8
  %15 = inttoptr i64 %14 to i32*
  store i32 -202116109, i32* %15
  %16 = ptrtoint i32* %3 to i64
  %17 = lshr i64 %16, 3
  %18 = add i64 %17, 2147450880
  %19 = inttoptr i64 %18 to i8*
  %20 = load i8, i8* %19
  %21 = icmp ne i8 %20, 0
  call void @llvm.dbg.declare(metadata i32* %3, metadata !23, metadata !28), !dbg !DILocation(scope: !4)
  br i1 %21, label %22, label %28

; <label>:22                                      ; preds = %entry
  %23 = and i64 %16, 7
  %24 = add i64 %23, 3
  %25 = trunc i64 %24 to i8
  %26 = icmp sge i8 %25, %20
  br i1 %26, label %27, label %28

; <label>:27                                      ; preds = %22
  call void @__asan_report_store4(i64 %16)
  call void asm sideeffect "", ""()
  unreachable

; <label>:28                                      ; preds = %22, %entry
  store i32 %0, i32* %3, align 4
  call void @llvm.dbg.declare(metadata %struct.A* %agg.result, metadata !24, metadata !DIExpression(DW_OP_deref)), !dbg !25
  call void @_ZN1AC1Ev(%struct.A* %agg.result), !dbg !25
  store i64 1172321806, i64* %4, !dbg !26
  %29 = inttoptr i64 %10 to i32*, !dbg !26
  store i32 0, i32* %29, !dbg !26
  %30 = add i64 %10, 4, !dbg !26
  %31 = inttoptr i64 %30 to i32*, !dbg !26
  store i32 0, i32* %31, !dbg !26
  %32 = add i64 %10, 8, !dbg !26
  %33 = inttoptr i64 %32 to i32*, !dbg !26
  store i32 0, i32* %33, !dbg !26
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_ZN1AC1Ev(%struct.A*) #2

define internal void @asan.module_ctor()  "stack-protector-buffer-size"="1" {
  call void @__asan_init_v3()
  %1 = load volatile i64, i64* @__asan_mapping_offset
  %2 = load volatile i64, i64* @__asan_mapping_scale
  ret void
}

declare void @__asan_init_v3()

declare void @__asan_report_load1(i64)

declare void @__asan_report_load2(i64)

declare void @__asan_report_load4(i64)

declare void @__asan_report_load8(i64)

declare void @__asan_report_load16(i64)

declare void @__asan_report_store1(i64)

declare void @__asan_report_store2(i64)

declare void @__asan_report_store4(i64)

declare void @__asan_report_store8(i64)

declare void @__asan_report_store16(i64)

declare void @__asan_report_load_n(i64, i64)

declare void @__asan_report_store_n(i64, i64)

declare void @__asan_handle_no_return()

declare i64 @__asan_stack_malloc(i64, i64)

declare void @__asan_stack_free(i64, i64, i64)

declare void @__asan_poison_stack_memory(i64, i64)

declare void @__asan_unpoison_stack_memory(i64, i64)

declare void @__asan_before_dynamic_init(i64)

declare void @__asan_after_dynamic_init()

declare void @__asan_register_globals(i64, i64)

declare void @__asan_unregister_globals(i64, i64)

attributes #0 = { sanitize_address uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "ssp-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "ssp-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !27}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "crash.cpp", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "func", linkageName: "_Z4funci", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "crash.cpp", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !21}
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 8, align: 8, file: !1, elements: !9)
!9 = !{!10, !15}
!10 = !DISubprogram(name: "A", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !8, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !8)
!15 = !DISubprogram(name: "A", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !8, type: !16)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !13, !18}
!18 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !19)
!19 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!21 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !{i32 2, !"Dwarf Version", i32 3}
!23 = !DILocalVariable(name: "", line: 6, arg: 1, scope: !4, file: !5, type: !21)
!24 = !DILocalVariable(name: "a", line: 7, scope: !4, file: !5, type: !8)
!25 = !DILocation(line: 7, scope: !4)
!26 = !DILocation(line: 8, scope: !4)
!27 = !{i32 1, !"Debug Info Version", i32 3}
!28 = !DIExpression(DW_OP_deref)
