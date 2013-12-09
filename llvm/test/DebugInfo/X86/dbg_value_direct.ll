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
define void @_Z4funci(%struct.A* noalias sret %agg.result, i32) #0 "stack-protector-buffer-size"="1" {
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
  %20 = load i8* %19
  %21 = icmp ne i8 %20, 0
  call void @llvm.dbg.declare(metadata !{i32* %3}, metadata !23)
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
  call void @llvm.dbg.declare(metadata !{%struct.A* %agg.result}, metadata !24), !dbg !25
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
declare void @llvm.dbg.declare(metadata, metadata) #1

declare void @_ZN1AC1Ev(%struct.A*) #2

define internal void @asan.module_ctor()  "stack-protector-buffer-size"="1" {
  call void @__asan_init_v3()
  %1 = load volatile i64* @__asan_mapping_offset
  %2 = load volatile i64* @__asan_mapping_scale
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

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/crash.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"crash.cpp", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func", metadata !"func", metadata !"_Z4funci", i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.A*, i32)* @_Z4funci, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/crash.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !21}
!8 = metadata !{i32 786451, metadata !1, null, metadata !"A", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [A] [line 1, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !15}
!10 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"A", metadata !"A", metadata !"", i32 2, metadata !11, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !14, i32 2} ; [ DW_TAG_subprogram ] [line 2] [A]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!14 = metadata !{i32 786468}
!15 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"A", metadata !"A", metadata !"", i32 3, metadata !16, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !20, i32 3} ; [ DW_TAG_subprogram ] [line 3] [A]
!16 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null, metadata !13, metadata !18}
!18 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !19} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from A]
!20 = metadata !{i32 786468}
!21 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!22 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!23 = metadata !{i32 786689, metadata !4, metadata !"", metadata !5, i32 16777222, metadata !21, i32 0, i32 0, i64 2} ; [ DW_TAG_arg_variable ] [line 6]
!24 = metadata !{i32 786688, metadata !4, metadata !"a", metadata !5, i32 7, metadata !8, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 7]
!25 = metadata !{i32 7, i32 0, metadata !4, null}
!26 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
