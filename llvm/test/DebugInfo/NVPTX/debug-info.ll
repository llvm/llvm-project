; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; // Bitcode int this test case is reduced version of compiled code below:
;__device__ inline void res(float x, float y, float *res) { *res = x + y; }
;
;__global__ void saxpy(int n, float a, float *x, float *y) {
;  int i = blockIdx.x * blockDim.x + threadIdx.x;
;  if (i < n)
;    res(a * x[i], y[i], &y[i]);
;}

; CHECK: .target sm_{{[0-9]+}}, debug

; CHECK: .visible .entry _Z5saxpyifPfS_(
; CHECK: .param .u32 {{.+}},
; CHECK: .param .f32 {{.+}},
; CHECK: .param .u64 {{.+}},
; CHECK: .param .u64 {{.+}}
; CHECK: )
; CHECK: {
; CHECK: .reg .pred      %p<2>;
; CHECK: .reg .f32       %f<5>;
; CHECK: .reg .b32       %r<6>;
; CHECK: .reg .b64       %rd<8>;
; CHECK: .loc [[DEBUG_INFO_CU:[0-9]+]] 5 0
; CHECK: ld.param.u32    %r{{.+}}, [{{.+}}];
; CHECK: .loc [[BUILTUIN_VARS_H:[0-9]+]] 78 180
; CHECK: mov.u32         %r{{.+}}, %ctaid.x;
; CHECK: .loc [[BUILTUIN_VARS_H]] 89 180
; CHECK: mov.u32         %r{{.+}}, %ntid.x;
; CHECK: .loc [[BUILTUIN_VARS_H]] 67 180
; CHECK: mov.u32         %r{{.+}}, %tid.x;
; CHECK: .loc [[DEBUG_INFO_CU]] 6 35
; CHECK: mad.lo.s32      %r{{.+}}, %r{{.+}}, %r{{.+}}, %r{{.+}};
; CHECK: .loc [[DEBUG_INFO_CU]] 7 9
; CHECK: setp.ge.s32     %p{{.+}}, %r{{.+}}, %r{{.+}};
; CHECK: .loc [[DEBUG_INFO_CU]] 7 7
; CHECK: @%p{{.+}} bra   [[BB:.+]];
; CHECK: ld.param.f32    %f{{.+}}, [{{.+}}];
; CHECK: ld.param.u64    %rd{{.+}}, [{{.+}}];
; CHECK: cvta.to.global.u64      %rd{{.+}}, %rd{{.+}};
; CHECK: ld.param.u64    %rd{{.+}}, [{{.+}}];
; CHECK: cvta.to.global.u64      %rd{{.+}}, %rd{{.+}};
; CHECK: mul.wide.u32    %rd{{.+}}, %r{{.+}}, 4;
; CHECK: add.s64         %rd{{.+}}, %rd{{.+}}, %rd{{.+}};
; CHECK: .loc [[DEBUG_INFO_CU]] 8 13
; CHECK: ld.global.f32   %f{{.+}}, [%rd{{.+}}];
; CHECK: add.s64         %rd{{.+}}, %rd{{.+}}, %rd{{.+}};
; CHECK: .loc [[DEBUG_INFO_CU]] 8 19
; CHECK: ld.global.f32   %f{{.+}}, [%rd{{.+}}];
; CHECK: .loc [[DEBUG_INFO_CU]] 3 82
; CHECK: fma.rn.f32      %f{{.+}}, %f{{.+}}, %f{{.+}}, %f{{.+}};
; CHECK: .loc [[DEBUG_INFO_CU]] 3 78
; CHECK: st.global.f32   [%rd{{.+}}], %f{{.+}};
; CHECK: [[BB]]:
; CHECK: .loc [[DEBUG_INFO_CU]] 9 1
; CHECK: ret;
; CHECK: }

; Function Attrs: nounwind
define void @_Z5saxpyifPfS_(i32 %n, float %a, float* nocapture readonly %x, float* nocapture %y) local_unnamed_addr #0 !dbg !566 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !570, metadata !DIExpression()), !dbg !575
  call void @llvm.dbg.value(metadata float %a, metadata !571, metadata !DIExpression()), !dbg !576
  call void @llvm.dbg.value(metadata float* %x, metadata !572, metadata !DIExpression()), !dbg !577
  call void @llvm.dbg.value(metadata float* %y, metadata !573, metadata !DIExpression()), !dbg !578
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !dbg !579, !range !616
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !dbg !617, !range !661
  %mul = mul nuw nsw i32 %1, %0, !dbg !662
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !dbg !663, !range !691
  %add = add nuw nsw i32 %mul, %2, !dbg !692
  call void @llvm.dbg.value(metadata i32 %add, metadata !574, metadata !DIExpression()), !dbg !693
  %cmp = icmp slt i32 %add, %n, !dbg !694
  br i1 %cmp, label %if.then, label %if.end, !dbg !696

if.then:                                          ; preds = %entry
  %3 = zext i32 %add to i64, !dbg !697
  %arrayidx = getelementptr inbounds float, float* %x, i64 %3, !dbg !697
  %4 = load float, float* %arrayidx, align 4, !dbg !697, !tbaa !698
  %mul3 = fmul contract float %4, %a, !dbg !702
  %arrayidx5 = getelementptr inbounds float, float* %y, i64 %3, !dbg !703
  %5 = load float, float* %arrayidx5, align 4, !dbg !703, !tbaa !698
  call void @llvm.dbg.value(metadata float %mul3, metadata !704, metadata !DIExpression()), !dbg !711
  call void @llvm.dbg.value(metadata float %5, metadata !709, metadata !DIExpression()), !dbg !713
  call void @llvm.dbg.value(metadata float* %arrayidx5, metadata !710, metadata !DIExpression()), !dbg !714
  %add.i = fadd contract float %mul3, %5, !dbg !715
  store float %add.i, float* %arrayidx5, align 4, !dbg !716, !tbaa !698
  br label %if.end, !dbg !717

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !718
}

; CHECK-DAG: .file {{[0-9]+}} "{{.*}}clang/include{{/|\\\\}}__clang_cuda_math_forward_declares.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/include{{/|\\\\}}mathcalls.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/lib/gcc/4.8/../../../../include/c++/4.8{{/|\\\\}}cmath"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/include{{/|\\\\}}stdlib.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/lib/gcc/4.8/../../../../include/c++/4.8{{/|\\\\}}cstdlib"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/include{{/|\\\\}}stdlib-float.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/include{{/|\\\\}}stdlib-bsearch.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}clang/include{{/|\\\\}}stddef.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/local/cuda/include{{/|\\\\}}math_functions.hpp"
; CHECK_DAG: .file {{[0-9]+}} "{{.*}}clang/include{{/|\\\\}}__clang_cuda_cmath.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/local/cuda/include{{/|\\\\}}device_functions.hpp"
; CHECK-DAG: .file [[DEBUG_INFO_CU]] "{{.*}}debug-info.cu"
; CHECK-DAG: .file [[BUILTUIN_VARS_H]] "{{.*}}clang/include{{/|\\\\}}__clang_cuda_builtin_vars.h"

; CHECK: .section .debug_abbrev
; CHECK-NEXT: {
; CHECK-NEXT: .b8 1                                // Abbreviation Code
; CHECK-NEXT: .b8 17                               // DW_TAG_compile_unit
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 37                               // DW_AT_producer
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 19                               // DW_AT_language
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 16                               // DW_AT_stmt_list
; CHECK-NEXT: .b8 6                                // DW_FORM_data4
; CHECK-NEXT: .b8 27                               // DW_AT_comp_dir
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 2                                // Abbreviation Code
; CHECK-NEXT: .b8 57                               // DW_TAG_namespace
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 3                                // Abbreviation Code
; CHECK-NEXT: .b8 8                                // DW_TAG_imported_declaration
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 24                               // DW_AT_import
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 4                                // Abbreviation Code
; CHECK-NEXT: .b8 8                                // DW_TAG_imported_declaration
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 24                               // DW_AT_import
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 5                                // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 6                                // Abbreviation Code
; CHECK-NEXT: .b8 5                                // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 7                                // Abbreviation Code
; CHECK-NEXT: .b8 36                               // DW_TAG_base_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 62                               // DW_AT_encoding
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 11                               // DW_AT_byte_size
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 8                                // Abbreviation Code
; CHECK-NEXT: .b8 15                               // DW_TAG_pointer_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 9                                // Abbreviation Code
; CHECK-NEXT: .b8 38                               // DW_TAG_const_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 10                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 11                               // Abbreviation Code
; CHECK-NEXT: .b8 22                               // DW_TAG_typedef
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 12                               // Abbreviation Code
; CHECK-NEXT: .b8 19                               // DW_TAG_structure_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 13                               // Abbreviation Code
; CHECK-NEXT: .b8 19                               // DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 11                               // DW_AT_byte_size
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 14                               // Abbreviation Code
; CHECK-NEXT: .b8 13                               // DW_TAG_member
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 56                               // DW_AT_data_member_location
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 15                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 135,1                            // DW_AT_noreturn
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 16                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 17                               // Abbreviation Code
; CHECK-NEXT: .b8 21                               // DW_TAG_subroutine_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 18                               // Abbreviation Code
; CHECK-NEXT: .b8 15                               // DW_TAG_pointer_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 19                               // Abbreviation Code
; CHECK-NEXT: .b8 38                               // DW_TAG_const_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 20                               // Abbreviation Code
; CHECK-NEXT: .b8 22                               // DW_TAG_typedef
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 21                               // Abbreviation Code
; CHECK-NEXT: .b8 21                               // DW_TAG_subroutine_type
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 22                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 135,1                            // DW_AT_noreturn
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 23                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 24                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 25                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 26                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 27                               // Abbreviation Code
; CHECK-NEXT: .b8 19                               // DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 11                               // DW_AT_byte_size
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 28                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 29                               // Abbreviation Code
; CHECK-NEXT: .b8 5                                // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 52                               // DW_AT_artificial
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 30                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 50                               // DW_AT_accessibility
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 31                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 50                               // DW_AT_accessibility
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 32                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 50                               // DW_AT_accessibility
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 33                               // Abbreviation Code
; CHECK-NEXT: .b8 16                               // DW_TAG_reference_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 34                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 71                               // DW_AT_specification
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 32                               // DW_AT_inline
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 35                               // Abbreviation Code
; CHECK-NEXT: .b8 19                               // DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 11                               // DW_AT_byte_size
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 36                               // Abbreviation Code
; CHECK-NEXT: .b8 13                               // DW_TAG_member
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 56                               // DW_AT_data_member_location
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 37                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 38                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 32                               // DW_AT_inline
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 39                               // Abbreviation Code
; CHECK-NEXT: .b8 5                                // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 40                               // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 64                               // DW_AT_frame_base
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 41                               // Abbreviation Code
; CHECK-NEXT: .b8 52                               // DW_TAG_variable
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 42                               // Abbreviation Code
; CHECK-NEXT: .b8 29                               // DW_TAG_inlined_subroutine
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 49                               // DW_AT_abstract_origin
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 88                               // DW_AT_call_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 89                               // DW_AT_call_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 43                               // Abbreviation Code
; CHECK-NEXT: .b8 29                               // DW_TAG_inlined_subroutine
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 49                               // DW_AT_abstract_origin
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 88                               // DW_AT_call_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 89                               // DW_AT_call_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 44                               // Abbreviation Code
; CHECK-NEXT: .b8 5                                // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 49                               // DW_AT_abstract_origin
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 0                                // EOM(3)
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_info
; CHECK-NEXT: {
; CHECK-NEXT: .b32 10030                           // Length of Unit
; CHECK-NEXT: .b8 2                                // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                // Abbrev [1] 0xb:0x2727 DW_TAG_compile_unit
; CHECK-NEXT: .b8 0                                // DW_AT_producer
; CHECK-NEXT: .b8 4                                // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 100,101,98,117,103,45,105,110,102,111,46,99,117 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                     // DW_AT_stmt_list
; CHECK-NEXT: .b8 47,115,111,109,101,47,100,105,114,101,99,116,111,114,121 // DW_AT_comp_dir
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 2                                // Abbrev [2] 0x41:0x588 DW_TAG_namespace
; CHECK-NEXT: .b8 115,116,100                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x46:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 202                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1481                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x4d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 203                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1525                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x54:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 204                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1563                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x5b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 205                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1594                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x62:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 206                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1623                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x69:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 207                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1654                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x70:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 208                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1683                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x77:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 209                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1720                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x7e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 210                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1751                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x85:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 211                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1780                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x8c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 212                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1809                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x93:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 213                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1852                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x9a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 214                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1879                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xa1:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 215                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1908                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xa8:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 216                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1935                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xaf:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 217                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1964                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xb6:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 218                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1991                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xbd:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 219                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2020                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xc4:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 220                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2051                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xcb:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 221                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2080                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xd2:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 222                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2115                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xd9:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 223                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2146                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xe0:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 224                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2185                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xe7:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 225                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2220                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xee:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 226                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2255                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xf5:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 227                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2290                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xfc:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 228                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2339                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x103:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 229                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2382                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x10a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 230                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2419                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x111:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 231                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2450                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x118:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 232                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2495                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x11f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 233                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2540                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x126:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 234                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2596                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x12d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 235                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2627                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x134:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 236                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2666                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x13b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 237                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2716                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x142:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 238                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2770                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x149:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 239                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2801                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x150:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 240                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2838                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x157:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 241                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2888                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x15e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 242                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2929                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x165:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 243                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2966                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x16c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 244                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2999                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x173:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 245                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3030                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x17a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 246                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3063                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x181:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 247                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3090                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x188:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 248                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3121                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x18f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 249                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3152                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x196:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 250                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3181                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x19d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 251                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3210                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1a4:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 252                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3241                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1ab:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 253                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3274                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1b2:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 254                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3309                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1b9:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 255                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3350                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1c0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 0                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3407                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1c8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3438                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1d0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 2                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3477                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1d8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3522                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1e0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 4                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3555                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1e8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3600                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1f0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3646                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x1f8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 7                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3675                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x200:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 8                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3706                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x208:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 9                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3747                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x210:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 10                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3786                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x218:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3821                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x220:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 12                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3848                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x228:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 13                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3877                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x230:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 14                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3906                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x238:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 15                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3933                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x240:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 16                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3962                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x248:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 17                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 3995                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x250:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 102                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4026                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x257:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 121                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4046                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x25e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 140                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4066                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x265:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 159                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4086                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x26c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 180                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4112                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x273:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 199                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4132                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x27a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 218                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4151                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x281:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 237                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4171                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x288:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 0                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4190                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x290:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 19                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4210                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x298:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 38                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4231                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2a0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4256                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2a8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 78                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4282                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2b0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 97                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4308                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2b8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 116                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4327                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2c0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 135                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4348                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2c8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 147                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4378                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2d0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 184                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4402                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2d8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 203                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4421                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2e0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 222                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4441                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2e8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 241                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4461                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x2f0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 3                                // DW_AT_decl_file
; CHECK-NEXT: .b8 4                                // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 4480                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x2f8:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 118                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4500                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x2ff:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 119                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4515                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x306:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 121                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4563                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x30d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 122                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4576                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x314:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 123                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4596                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x31b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 129                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4625                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x322:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 130                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4645                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x329:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 131                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4666                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x330:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 132                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4687                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x337:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 133                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4815                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x33e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 134                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4843                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x345:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 135                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4868                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x34c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 136                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4886                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x353:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 137                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4903                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x35a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 138                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4931                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x361:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 139                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4952                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x368:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 140                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4978                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x36f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 142                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5001                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x376:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 143                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5028                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x37d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 144                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5079                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x384:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 146                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5112                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x38b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 152                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5145                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x392:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 153                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5160                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x399:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 154                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5189                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3a0:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 155                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5223                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3a7:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 156                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5255                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3ae:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 157                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5287                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3b5:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 158                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5320                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3bc:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 160                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5343                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3c3:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 161                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5388                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3ca:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 241                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5536                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3d1:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 243                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5585                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3d8:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 245                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5604                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3df:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 246                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5490                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3e6:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 247                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5626                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3ed:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 249                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5653                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3f4:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 250                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5768                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x3fb:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 251                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5675                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x402:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 252                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5708                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x409:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 253                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5795                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x410:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 149                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 5838                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x418:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 150                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 5870                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x420:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 151                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 5904                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x428:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 152                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 5936                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x430:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 153                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 5970                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x438:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 154                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6010                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x440:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 155                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6042                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x448:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 156                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6076                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x450:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 157                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6108                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x458:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 158                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6140                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x460:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 159                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6186                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x468:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 160                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6216                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x470:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 161                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6248                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x478:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 162                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6280                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x480:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 163                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6310                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x488:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 164                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6342                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x490:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 165                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6372                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x498:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 166                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6406                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4a0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 167                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6438                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4a8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 168                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6476                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4b0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 169                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6510                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4b8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 170                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6552                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4c0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 171                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6590                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4c8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 172                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6628                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4d0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 173                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6666                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4d8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 174                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6707                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4e0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 175                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6747                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4e8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 176                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6781                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4f0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 177                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6821                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x4f8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 178                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6857                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x500:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 179                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6893                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x508:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 180                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6931                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x510:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 181                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6965                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x518:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 182                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 6999                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x520:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 183                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7031                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x528:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 184                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7063                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x530:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 185                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7093                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x538:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 186                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7127                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x540:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 187                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7163                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x548:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 188                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7202                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x550:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 189                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7245                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x558:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 190                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7294                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x560:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 191                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7330                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x568:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 192                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7379                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x570:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 193                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7428                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x578:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 194                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7460                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x580:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 195                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7494                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x588:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 196                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7538                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x590:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 197                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7580                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x598:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 198                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7610                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x5a0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 199                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7642                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x5a8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 200                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7674                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x5b0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 201                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7704                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x5b8:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 202                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7736                            // DW_AT_import
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x5c0:0x8 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 10                               // DW_AT_decl_file
; CHECK-NEXT: .b8 203                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 7772                            // DW_AT_import
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x5c9:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,97,98,115,120        // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,98,115                        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 44                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x5de:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x5e4:0x11 DW_TAG_base_type
; CHECK-NEXT: .b8 108,111,110,103,32,108,111,110,103,32,105,110,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 8                                // DW_AT_byte_size
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x5f5:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,97,99,111,115,102    // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,99,111,115                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 46                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x60c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x612:0x9 DW_TAG_base_type
; CHECK-NEXT: .b8 102,108,111,97,116               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x61b:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,99,111,115,104,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,99,111,115,104                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 48                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x634:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x63a:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,97,115,105,110,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,115,105,110                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 50                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x651:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x657:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,115,105,110,104,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,115,105,110,104               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 52                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x670:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x676:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,97,116,97,110,102    // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,116,97,110                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 56                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x68d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x693:0x25 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,116,97,110,50,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,116,97,110,50                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 54                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x6ad:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x6b2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x6b8:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,116,97,110,104,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,116,97,110,104                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 58                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x6d1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x6d7:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,99,98,114,116,102    // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,98,114,116                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 60                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x6ee:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x6f4:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,99,101,105,108,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,101,105,108                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 62                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x70b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x711:0x2b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,56,99,111,112,121,115,105,103,110,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,111,112,121,115,105,103,110   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 64                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x731:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x736:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x73c:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,99,111,115,102       // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,111,115                       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 66                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x751:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x757:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,99,111,115,104,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,111,115,104                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 68                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x76e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x774:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,101,114,102,102      // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,114,102                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 72                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x789:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x78f:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,101,114,102,99,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,114,102,99                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 70                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x7a6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x7ac:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,101,120,112,102      // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,120,112                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 76                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x7c1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x7c7:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,101,120,112,50,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,120,112,50                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 74                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x7de:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x7e4:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,101,120,112,109,49,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,120,112,109,49               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 78                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x7fd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x803:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,102,97,98,115,102    // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,97,98,115                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 80                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x81a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x820:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,102,100,105,109,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,100,105,109                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 82                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x838:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x83d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x843:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,108,111,111,114,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,108,111,111,114              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 84                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x85c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x862:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,102,109,97,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,97                       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 86                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x879:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x87e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x883:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x889:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,102,109,97,120,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,97,120                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 88                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8a1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8a6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x8ac:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,102,109,105,110,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,105,110                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 90                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8c4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8c9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x8cf:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,102,109,111,100,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,111,100                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 92                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8e7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8ec:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x8f2:0x2a DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,48,102,112,99,108,97,115,115,105,102,121,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,112,99,108,97,115,115,105,102,121 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 94                               // DW_AT_decl_line
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x916:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x91c:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105,110,116                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x923:0x26 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,114,101,120,112,102,80,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,114,101,120,112              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 96                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x93e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x943:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2377                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x949:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x94e:0x25 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,104,121,112,111,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 104,121,112,111,116              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 98                               // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x968:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x96d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x973:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,105,108,111,103,98,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,108,111,103,98               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 100                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x98c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x992:0x25 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,56,105,115,102,105,110,105,116,101,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,102,105,110,105,116,101  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 102                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x9b1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x9b7:0x8 DW_TAG_base_type
; CHECK-NEXT: .b8 98,111,111,108                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_encoding
; CHECK-NEXT: .b8 1                                // DW_AT_byte_size
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x9bf:0x2d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,57,105,115,103,114,101,97,116,101,114,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,103,114,101,97,116,101,114 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 106                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x9e1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x9e6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x9ec:0x38 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,52,105,115,103,114,101,97,116,101,114,101,113,117,97,108,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,103,114,101,97,116,101,114,101,113,117,97,108 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 105                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa19:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa1e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xa24:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,105,115,105,110,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,105,110,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 108                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa3d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xa43:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,105,115,108,101,115,115,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,108,101,115,115          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 112                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa5f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa64:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xa6a:0x32 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,49,105,115,108,101,115,115,101,113,117,97,108,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,108,101,115,115,101,113,117,97,108 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 111                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa91:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xa96:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xa9c:0x36 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,51,105,115,108,101,115,115,103,114,101,97,116,101,114,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,108,101,115,115,103,114,101,97,116,101,114 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 114                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xac7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xacc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xad2:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,105,115,110,97,110,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,110,97,110               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 116                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xaeb:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xaf1:0x25 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,56,105,115,110,111,114,109,97,108,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,110,111,114,109,97,108   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 118                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xb10:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xb16:0x32 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,49,105,115,117,110,111,114,100,101,114,101,100,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,115,117,110,111,114,100,101,114,101,100 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 120                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xb3d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xb42:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xb48:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,108,97,98,115,108    // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,97,98,115                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 121                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xb5f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0xb65:0xc DW_TAG_base_type
; CHECK-NEXT: .b8 108,111,110,103,32,105,110,116   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 8                                // DW_AT_byte_size
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xb71:0x25 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,100,101,120,112,102,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,100,101,120,112              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 123                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xb8b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xb90:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xb96:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,103,97,109,109,97,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,103,97,109,109,97            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 125                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xbb1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xbb7:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,108,97,98,115,120 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,108,97,98,115                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 126                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xbd0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xbd6:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,108,114,105,110,116,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,108,114,105,110,116          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 128                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xbf1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xbf7:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,108,111,103,102      // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 138                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xc0c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xc12:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,111,103,49,48,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,49,48                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 130                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xc2b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xc31:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,111,103,49,112,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,49,112               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 132                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xc4a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xc50:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,108,111,103,50,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,50                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 134                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xc67:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xc6d:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,108,111,103,98,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,98                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 136                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xc84:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xc8a:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,114,105,110,116,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,114,105,110,116              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 140                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xca3:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xca9:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,114,111,117,110,100,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,114,111,117,110,100          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 142                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xcc4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xcca:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,108,108,114,111,117,110,100,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,108,114,111,117,110,100      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 143                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xce7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xced:0x24 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,109,111,100,102,102,80,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 109,111,100,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 145                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xd06:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xd0b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3345                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0xd11:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xd16:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,110,97,110,80,75,99  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,97,110                       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 146                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xd2d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0xd33:0xa DW_TAG_base_type
; CHECK-NEXT: .b8 100,111,117,98,108,101           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_encoding
; CHECK-NEXT: .b8 8                                // DW_AT_byte_size
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0xd3d:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 3394                            // DW_AT_type
; CHECK-NEXT: .b8 9                                // Abbrev [9] 0xd42:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 3399                            // DW_AT_type
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0xd47:0x8 DW_TAG_base_type
; CHECK-NEXT: .b8 99,104,97,114                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 8                                // DW_AT_encoding
; CHECK-NEXT: .b8 1                                // DW_AT_byte_size
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xd4f:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,110,97,110,102,80,75,99 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,97,110,102                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 147                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xd68:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xd6e:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,57,110,101,97,114,98,121,105,110,116,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,101,97,114,98,121,105,110,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 149                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xd8f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xd95:0x2d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,57,110,101,120,116,97,102,116,101,114,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,101,120,116,97,102,116,101,114 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 151                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xdb7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xdbc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xdc2:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,112,111,119,102,105  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 112,111,119                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 155                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xdd8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xddd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xde3:0x2d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,57,114,101,109,97,105,110,100,101,114,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,101,109,97,105,110,100,101,114 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 157                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe05:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe0a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xe10:0x2e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,114,101,109,113,117,111,102,102,80,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,101,109,113,117,111          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 159                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe2e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe33:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe38:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2377                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xe3e:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,114,105,110,116,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,105,110,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 161                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe55:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xe5b:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,114,111,117,110,100,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,111,117,110,100              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 163                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe74:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xe7a:0x29 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,115,99,97,108,98,108,110,102,108 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,99,97,108,98,108,110         // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 165                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe98:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xe9d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xea3:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,115,99,97,108,98,110,102,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,99,97,108,98,110             // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 167                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xebf:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xec4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xeca:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,115,105,103,110,98,105,116,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,105,103,110,98,105,116       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 169                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2487                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xee7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xeed:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,115,105,110,102      // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,105,110                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 171                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xf02:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xf08:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,115,105,110,104,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,105,110,104                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 173                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xf1f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xf25:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,115,113,114,116,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,113,114,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 175                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xf3c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xf42:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,51,116,97,110,102       // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,97,110                       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 177                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xf57:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xf5d:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,116,97,110,104,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,97,110,104                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 179                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xf74:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xf7a:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,116,103,97,109,109,97,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,103,97,109,109,97            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 181                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xf95:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0xf9b:0x1f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,116,114,117,110,99,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,114,117,110,99               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 183                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xfb4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0xfba:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,99,111,115                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 54                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xfc8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0xfce:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,115,105,110                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 56                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xfdc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0xfe2:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,97,110                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 58                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0xff0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0xff6:0x1a DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,97,110,50                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 60                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1005:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x100a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1010:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 99,101,105,108                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 178                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x101e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1024:0x13 DW_TAG_subprogram
; CHECK-NEXT: .b8 99,111,115                       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 63                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1031:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1037:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 99,111,115,104                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 72                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1045:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x104b:0x13 DW_TAG_subprogram
; CHECK-NEXT: .b8 101,120,112                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 100                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1058:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x105e:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 102,97,98,115                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 181                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x106c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1072:0x15 DW_TAG_subprogram
; CHECK-NEXT: .b8 102,108,111,111,114              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 184                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1081:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1087:0x19 DW_TAG_subprogram
; CHECK-NEXT: .b8 102,109,111,100                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 187                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1095:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x109a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x10a0:0x1a DW_TAG_subprogram
; CHECK-NEXT: .b8 102,114,101,120,112              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 103                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x10af:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x10b4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2377                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x10ba:0x1a DW_TAG_subprogram
; CHECK-NEXT: .b8 108,100,101,120,112              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 106                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x10c9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x10ce:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x10d4:0x13 DW_TAG_subprogram
; CHECK-NEXT: .b8 108,111,103                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 109                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x10e1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x10e7:0x15 DW_TAG_subprogram
; CHECK-NEXT: .b8 108,111,103,49,48                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 112                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x10f6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x10fc:0x19 DW_TAG_subprogram
; CHECK-NEXT: .b8 109,111,100,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 115                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x110a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x110f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4373                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x1115:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x111a:0x18 DW_TAG_subprogram
; CHECK-NEXT: .b8 112,111,119                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 153                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1127:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x112c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1132:0x13 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,105,110                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 65                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x113f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1145:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,105,110,104                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 74                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1153:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1159:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,113,114,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 156                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1167:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x116d:0x13 DW_TAG_subprogram
; CHECK-NEXT: .b8 116,97,110                       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 67                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x117a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1180:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 116,97,110,104                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 76                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x118e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 11                               // Abbrev [11] 0x1194:0xd DW_TAG_typedef
; CHECK-NEXT: .b32 4513                            // DW_AT_type
; CHECK-NEXT: .b8 100,105,118,95,116               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 101                              // DW_AT_decl_line
; CHECK-NEXT: .b8 12                               // Abbrev [12] 0x11a1:0x2 DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 11                               // Abbrev [11] 0x11a3:0xe DW_TAG_typedef
; CHECK-NEXT: .b32 4529                            // DW_AT_type
; CHECK-NEXT: .b8 108,100,105,118,95,116           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 109                              // DW_AT_decl_line
; CHECK-NEXT: .b8 13                               // Abbrev [13] 0x11b1:0x22 DW_TAG_structure_type
; CHECK-NEXT: .b8 16                               // DW_AT_byte_size
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 105                              // DW_AT_decl_line
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x11b5:0xf DW_TAG_member
; CHECK-NEXT: .b8 113,117,111,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 107                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x11c4:0xe DW_TAG_member
; CHECK-NEXT: .b8 114,101,109                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 108                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 8
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 15                               // Abbrev [15] 0x11d3:0xd DW_TAG_subprogram
; CHECK-NEXT: .b8 97,98,111,114,116                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 1                                // DW_AT_noreturn
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x11e0:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,98,115                        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 7                                // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x11ee:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x11f4:0x17 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,101,120,105,116           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 7                                // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1205:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4619                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x120b:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 4624                            // DW_AT_type
; CHECK-NEXT: .b8 17                               // Abbrev [17] 0x1210:0x1 DW_TAG_subroutine_type
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1211:0x14 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,111,102                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 6                                // DW_AT_decl_file
; CHECK-NEXT: .b8 26                               // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x121f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1225:0x15 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,111,105                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 22                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1234:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x123a:0x15 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,111,108                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 27                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1249:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x124f:0x2b DW_TAG_subprogram
; CHECK-NEXT: .b8 98,115,101,97,114,99,104         // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 7                                // DW_AT_decl_file
; CHECK-NEXT: .b8 20                               // DW_AT_decl_line
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1260:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4731                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1265:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4731                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x126a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x126f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1274:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4772                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 18                               // Abbrev [18] 0x127a:0x1 DW_TAG_pointer_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x127b:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 4736                            // DW_AT_type
; CHECK-NEXT: .b8 19                               // Abbrev [19] 0x1280:0x1 DW_TAG_const_type
; CHECK-NEXT: .b8 11                               // Abbrev [11] 0x1281:0xe DW_TAG_typedef
; CHECK-NEXT: .b32 4751                            // DW_AT_type
; CHECK-NEXT: .b8 115,105,122,101,95,116           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 8                                // DW_AT_decl_file
; CHECK-NEXT: .b8 62                               // DW_AT_decl_line
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x128f:0x15 DW_TAG_base_type
; CHECK-NEXT: .b8 108,111,110,103,32,117,110,115,105,103,110,101,100,32,105,110,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 7                                // DW_AT_encoding
; CHECK-NEXT: .b8 8                                // DW_AT_byte_size
; CHECK-NEXT: .b8 20                               // Abbrev [20] 0x12a4:0x16 DW_TAG_typedef
; CHECK-NEXT: .b32 4794                            // DW_AT_type
; CHECK-NEXT: .b8 95,95,99,111,109,112,97,114,95,102,110,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 230                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x12ba:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 4799                            // DW_AT_type
; CHECK-NEXT: .b8 21                               // Abbrev [21] 0x12bf:0x10 DW_TAG_subroutine_type
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x12c4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4731                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x12c9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4731                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x12cf:0x1c DW_TAG_subprogram
; CHECK-NEXT: .b8 99,97,108,108,111,99             // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 212                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x12e0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x12e5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x12eb:0x19 DW_TAG_subprogram
; CHECK-NEXT: .b8 100,105,118                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 21                               // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 4500                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x12f9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x12fe:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 22                               // Abbrev [22] 0x1304:0x12 DW_TAG_subprogram
; CHECK-NEXT: .b8 101,120,105,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 31                               // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 1                                // DW_AT_noreturn
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1310:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 23                               // Abbrev [23] 0x1316:0x11 DW_TAG_subprogram
; CHECK-NEXT: .b8 102,114,101,101                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 227                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1321:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1327:0x17 DW_TAG_subprogram
; CHECK-NEXT: .b8 103,101,116,101,110,118          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 52                               // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 4926                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1338:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x133e:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 3399                            // DW_AT_type
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1343:0x15 DW_TAG_subprogram
; CHECK-NEXT: .b8 108,97,98,115                    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 8                                // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1352:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1358:0x1a DW_TAG_subprogram
; CHECK-NEXT: .b8 108,100,105,118                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 23                               // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 4515                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1367:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x136c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1372:0x17 DW_TAG_subprogram
; CHECK-NEXT: .b8 109,97,108,108,111,99            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 210                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1383:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1389:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 109,98,108,101,110               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 95                               // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1399:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x139e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x13a4:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 109,98,115,116,111,119,99,115    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 106                              // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x13b7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5063                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x13bc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x13c1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x13c7:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 5068                            // DW_AT_type
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x13cc:0xb DW_TAG_base_type
; CHECK-NEXT: .b8 119,99,104,97,114,95,116         // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x13d7:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 109,98,116,111,119,99            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 98                               // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x13e8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5063                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x13ed:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x13f2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 23                               // Abbrev [23] 0x13f8:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 113,115,111,114,116              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 253                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1404:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1409:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x140e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1413:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4772                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 24                               // Abbrev [24] 0x1419:0xf DW_TAG_subprogram
; CHECK-NEXT: .b8 114,97,110,100                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 118                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1428:0x1d DW_TAG_subprogram
; CHECK-NEXT: .b8 114,101,97,108,108,111,99        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 224                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x143a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4730                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x143f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 23                               // Abbrev [23] 0x1445:0x12 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,114,97,110,100               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 120                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1451:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x1457:0x10 DW_TAG_base_type
; CHECK-NEXT: .b8 117,110,115,105,103,110,101,100,32,105,110,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 7                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1467:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,100          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 164                              // DW_AT_decl_line
; CHECK-NEXT: .b32 3379                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1477:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x147c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x1482:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 4926                            // DW_AT_type
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1487:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,108          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 183                              // DW_AT_decl_line
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1497:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x149c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14a1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x14a7:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,117,108      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 187                              // DW_AT_decl_line
; CHECK-NEXT: .b32 4751                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14b8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14bd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14c2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x14c8:0x17 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,121,115,116,101,109          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 205                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14d9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x14df:0x23 DW_TAG_subprogram
; CHECK-NEXT: .b8 119,99,115,116,111,109,98,115    // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 109                              // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14f2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4926                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14f7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5378                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x14fc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4737                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x1502:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 5383                            // DW_AT_type
; CHECK-NEXT: .b8 9                                // Abbrev [9] 0x1507:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 5068                            // DW_AT_type
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x150c:0x1c DW_TAG_subprogram
; CHECK-NEXT: .b8 119,99,116,111,109,98            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 102                              // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x151d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 4926                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1522:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5068                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 2                                // Abbrev [2] 0x1528:0x78 DW_TAG_namespace
; CHECK-NEXT: .b8 95,95,103,110,117,95,99,120,120  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1533:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 201                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5536                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x153a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 207                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5585                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1541:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 211                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5604                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1548:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 217                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5626                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x154f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 228                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5653                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1556:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 229                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5675                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x155d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 230                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5708                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x1564:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 232                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5768                            // DW_AT_import
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x156b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 233                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5795                            // DW_AT_import
; CHECK-NEXT: .b8 25                               // Abbrev [25] 0x1572:0x2d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,57,95,95,103,110,117,95,99,120,120,51,100,105,118,69,120,120 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 100,105,118                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_decl_file
; CHECK-NEXT: .b8 214                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5536                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1594:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1599:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 11                               // Abbrev [11] 0x15a0:0xf DW_TAG_typedef
; CHECK-NEXT: .b32 5551                            // DW_AT_type
; CHECK-NEXT: .b8 108,108,100,105,118,95,116       // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 121                              // DW_AT_decl_line
; CHECK-NEXT: .b8 13                               // Abbrev [13] 0x15af:0x22 DW_TAG_structure_type
; CHECK-NEXT: .b8 16                               // DW_AT_byte_size
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 117                              // DW_AT_decl_line
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x15b3:0xf DW_TAG_member
; CHECK-NEXT: .b8 113,117,111,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 119                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x15c2:0xe DW_TAG_member
; CHECK-NEXT: .b8 114,101,109                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 120                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 8
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 22                               // Abbrev [22] 0x15d1:0x13 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,69,120,105,116                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 45                               // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 1                                // DW_AT_noreturn
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x15de:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x15e4:0x16 DW_TAG_subprogram
; CHECK-NEXT: .b8 108,108,97,98,115                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 12                               // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x15f4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x15fa:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 108,108,100,105,118              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 29                               // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 5536                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x160a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x160f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 16                               // Abbrev [16] 0x1615:0x16 DW_TAG_subprogram
; CHECK-NEXT: .b8 97,116,111,108,108               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 36                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1625:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x162b:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,108,108      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 209                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x163c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1641:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1646:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x164c:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,117,108,108  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 214                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5742                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x165e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1663:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1668:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x166e:0x1a DW_TAG_base_type
; CHECK-NEXT: .b8 108,111,110,103,32,108,111,110,103,32,117,110,115,105,103,110,101,100,32,105,110,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 7                                // DW_AT_encoding
; CHECK-NEXT: .b8 8                                // DW_AT_byte_size
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x1688:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 172                              // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1698:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x169d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 10                               // Abbrev [10] 0x16a3:0x1c DW_TAG_subprogram
; CHECK-NEXT: .b8 115,116,114,116,111,108,100      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_decl_file
; CHECK-NEXT: .b8 175                              // DW_AT_decl_line
; CHECK-NEXT: .b32 5823                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x16b4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3389                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x16b9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5250                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x16bf:0xf DW_TAG_base_type
; CHECK-NEXT: .b8 108,111,110,103,32,100,111,117,98,108,101 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_encoding
; CHECK-NEXT: .b8 8                                // DW_AT_byte_size
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x16ce:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,99,111,115,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,99,111,115,102                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 62                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x16e8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x16ee:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,97,99,111,115,104,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,99,111,115,104,102            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 90                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x170a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1710:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,115,105,110,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,115,105,110,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 57                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x172a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1730:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,97,115,105,110,104,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,115,105,110,104,102           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 95                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x174c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1752:0x28 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,97,116,97,110,50,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,116,97,110,50,102             // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 47                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x176f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1774:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x177a:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,97,116,97,110,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,116,97,110,102                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 52                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1794:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x179a:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,97,116,97,110,104,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97,116,97,110,104,102            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 100                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x17b6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x17bc:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,99,98,114,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,98,114,116,102                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 150                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x17d6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x17dc:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,99,101,105,108,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,101,105,108,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 155                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x17f6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x17fc:0x2e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,57,99,111,112,121,115,105,103,110,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,111,112,121,115,105,103,110,102 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 165                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x181f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1824:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x182a:0x1e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,99,111,115,102,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,111,115,102                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 219                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1842:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1848:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,99,111,115,104,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 99,111,115,104,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 32                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1862:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1868:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,101,114,102,99,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,114,102,99,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 210                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1882:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1888:0x1e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,101,114,102,102,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,114,102,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 200                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x18a0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x18a6:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,101,120,112,50,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,120,112,50,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 145                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x18c0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x18c6:0x1e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,101,120,112,102,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,120,112,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 14                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x18de:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x18e4:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,101,120,112,109,49,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 101,120,112,109,49,102           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 105                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1900:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1906:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,97,98,115,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,97,98,115,102                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 95                               // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1920:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1926:0x26 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,100,105,109,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,100,105,109,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 80                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1941:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1946:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x194c:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,102,108,111,111,114,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,108,111,111,114,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1968:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x196e:0x2a DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,102,109,97,102,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,97,102                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 32                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1988:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x198d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1992:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1998:0x26 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,109,97,120,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,97,120,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 110                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x19b3:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x19b8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x19be:0x26 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,109,105,110,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,105,110,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 105                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x19d9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x19de:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x19e4:0x26 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,102,109,111,100,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,109,111,100,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 17                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x19ff:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a04:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1a0a:0x29 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,102,114,101,120,112,102,102,80,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 102,114,101,120,112,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 7                                // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a28:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a2d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2377                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1a33:0x28 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,104,121,112,111,116,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 104,121,112,111,116,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 110                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a50:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a55:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1a5b:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,105,108,111,103,98,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 105,108,111,103,98,102           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a77:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1a7d:0x28 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,100,101,120,112,102,102,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,100,101,120,112,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 240                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a9a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1a9f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1aa5:0x24 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,108,103,97,109,109,97,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,103,97,109,109,97,102        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 235                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1ac3:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1ac9:0x24 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,108,108,114,105,110,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,108,114,105,110,116,102      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 125                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1ae7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1aed:0x26 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,56,108,108,114,111,117,110,100,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,108,114,111,117,110,100,102  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 66                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1508                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1b0d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1b13:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,111,103,49,48,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,49,48,102            // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 76                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1b2f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1b35:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,111,103,49,112,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,49,112,102           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1b51:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1b57:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,111,103,50,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,50,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1b71:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1b77:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,108,111,103,98,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,98,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 90                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1b91:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1b97:0x1e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,108,111,103,102,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,111,103,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 67                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1baf:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1bb5:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,108,114,105,110,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,114,105,110,116,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 116                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1bd1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1bd7:0x24 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,108,114,111,117,110,100,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 108,114,111,117,110,100,102      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 71                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1bf5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1bfb:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,109,111,100,102,102,102,80,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 109,111,100,102,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 12                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c17:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c1c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 3345                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1c22:0x2b DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,48,110,101,97,114,98,121,105,110,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,101,97,114,98,121,105,110,116,102 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 130                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c47:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1c4d:0x31 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,48,110,101,120,116,97,102,116,101,114,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,101,120,116,97,102,116,101,114,102 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 194                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c73:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c78:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1c7e:0x24 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,112,111,119,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 112,111,119,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 47                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c97:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1c9c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1ca2:0x31 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,49,48,114,101,109,97,105,110,100,101,114,102,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,101,109,97,105,110,100,101,114,102 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 22                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1cc8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1ccd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1cd3:0x31 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,114,101,109,113,117,111,102,102,102,80,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,101,109,113,117,111,102      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 27                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1cf4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1cf9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1cfe:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2377                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1d04:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,114,105,110,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,105,110,116,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 111                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1d1e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1d24:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,114,111,117,110,100,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,111,117,110,100,102          // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 61                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1d40:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1d46:0x2c DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,56,115,99,97,108,98,108,110,102,102,108 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,99,97,108,98,108,110,102     // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 250                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1d67:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1d6c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2917                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1d72:0x2a DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,115,99,97,108,98,110,102,102,105 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,99,97,108,98,110,102         // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 245                              // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1d91:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1d96:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1d9c:0x1e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,115,105,110,102,102  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,105,110,102                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 210                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1db4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1dba:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,115,105,110,104,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,105,110,104,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 37                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1dd4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1dda:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,115,113,114,116,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,113,114,116,102              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 139                              // DW_AT_decl_line
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1df4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1dfa:0x1e DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,52,116,97,110,102,102   // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,97,110,102                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 252                              // DW_AT_decl_line
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1e12:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1e18:0x20 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,53,116,97,110,104,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,97,110,104,102               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 42                               // DW_AT_decl_line
; CHECK-NEXT: .b8 5
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1e32:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1e38:0x24 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,55,116,103,97,109,109,97,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,103,97,109,109,97,102        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 9                                // DW_AT_decl_file
; CHECK-NEXT: .b8 56                               // DW_AT_decl_line
; CHECK-NEXT: .b8 6
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1e56:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 26                               // Abbrev [26] 0x1e5c:0x22 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,76,54,116,114,117,110,99,102,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,114,117,110,99,102           // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                               // DW_AT_decl_file
; CHECK-NEXT: .b8 150                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x1e78:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 27                               // Abbrev [27] 0x1e7e:0x22a DW_TAG_structure_type
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_byte_size
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 77                               // DW_AT_decl_line
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x1e9c:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116,49,55,95,95,102,101,116,99,104,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 98,117,105,108,116,105,110,95,120,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,120 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 78                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x1eeb:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116,49,55,95,95,102,101,116,99,104,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 98,117,105,108,116,105,110,95,121,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,121 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 79                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x1f3a:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116,49,55,95,95,102,101,116,99,104,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 98,117,105,108,116,105,110,95,122,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,122 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 80                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 25                               // Abbrev [25] 0x1f89:0x49 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116,99,118,53,117,105,110,116,51,69 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,32,117,105,110,116,51 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 83                               // DW_AT_decl_line
; CHECK-NEXT: .b32 8360                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x1fcb:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8407                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 30                               // Abbrev [30] 0x1fd2:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x1ff2:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8417                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 30                               // Abbrev [30] 0x1ff9:0x2c DW_TAG_subprogram
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x2019:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8417                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x201f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8422                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 31                               // Abbrev [31] 0x2025:0x43 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116,97,83,69,82,75,83,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,61 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x205c:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8407                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x2062:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8422                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 32                               // Abbrev [32] 0x2068:0x3f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,73,100,120,95,116,97,100,69,118 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,38 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 85                               // DW_AT_decl_line
; CHECK-NEXT: .b32 8427                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x20a0:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 8407                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 27                               // Abbrev [27] 0x20a8:0x2f DW_TAG_structure_type
; CHECK-NEXT: .b8 117,105,110,116,51               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_byte_size
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 190                              // DW_AT_decl_line
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x20b2:0xc DW_TAG_member
; CHECK-NEXT: .b8 120                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 192                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x20be:0xc DW_TAG_member
; CHECK-NEXT: .b8 121                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 192                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 14                               // Abbrev [14] 0x20ca:0xc DW_TAG_member
; CHECK-NEXT: .b8 122                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 192                              // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 8
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x20d7:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 8412                            // DW_AT_type
; CHECK-NEXT: .b8 9                                // Abbrev [9] 0x20dc:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 7806                            // DW_AT_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x20e1:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 7806                            // DW_AT_type
; CHECK-NEXT: .b8 33                               // Abbrev [33] 0x20e6:0x5 DW_TAG_reference_type
; CHECK-NEXT: .b32 8412                            // DW_AT_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x20eb:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 7806                            // DW_AT_type
; CHECK-NEXT: .b8 34                               // Abbrev [34] 0x20f0:0x6 DW_TAG_subprogram
; CHECK-NEXT: .b32 7836                            // DW_AT_specification
; CHECK-NEXT: .b8 1                                // DW_AT_inline
; CHECK-NEXT: .b8 27                               // Abbrev [27] 0x20f6:0x228 DW_TAG_structure_type
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_byte_size
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 88                               // DW_AT_decl_line
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x2114:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116,49,55,95,95,102,101,116,99,104,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 98,117,105,108,116,105,110,95,120,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,120 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 89                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x2163:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116,49,55,95,95,102,101,116,99,104,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 98,117,105,108,116,105,110,95,121,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,121 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 90                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x21b2:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116,49,55,95,95,102,101,116,99,104,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 98,117,105,108,116,105,110,95,122,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,122 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 91                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 25                               // Abbrev [25] 0x2201:0x47 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116,99,118,52,100,105,109,51,69,118 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,32,100,105,109,51 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 94                               // DW_AT_decl_line
; CHECK-NEXT: .b32 8990                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x2241:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9166                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 30                               // Abbrev [30] 0x2248:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 96                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x2268:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9176                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 30                               // Abbrev [30] 0x226f:0x2c DW_TAG_subprogram
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 96                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x228f:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9176                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x2295:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9181                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 31                               // Abbrev [31] 0x229b:0x43 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116,97,83,69,82,75,83,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,61 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 96                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x22d2:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9166                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x22d8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9181                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 32                               // Abbrev [32] 0x22de:0x3f DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,53,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,98,108,111,99,107,68,105,109,95,116,97,100,69,118 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,38 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 96                               // DW_AT_decl_line
; CHECK-NEXT: .b32 9186                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x2316:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9166                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 35                               // Abbrev [35] 0x231e:0x9d DW_TAG_structure_type
; CHECK-NEXT: .b8 100,105,109,51                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_byte_size
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 161                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 36                               // Abbrev [36] 0x2328:0xd DW_TAG_member
; CHECK-NEXT: .b8 120                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 163                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 36                               // Abbrev [36] 0x2335:0xd DW_TAG_member
; CHECK-NEXT: .b8 121                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 163                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 36                               // Abbrev [36] 0x2342:0xd DW_TAG_member
; CHECK-NEXT: .b8 122                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 163                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 8
; CHECK-NEXT: .b8 23                               // Abbrev [23] 0x234f:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 100,105,109,51                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 165                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x235a:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9147                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x2360:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x2365:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x236a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 23                               // Abbrev [23] 0x2370:0x17 DW_TAG_subprogram
; CHECK-NEXT: .b8 100,105,109,51                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 166                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x237b:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9147                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x2381:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9152                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 37                               // Abbrev [37] 0x2387:0x33 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,52,100,105,109,51,99,118,53,117,105,110,116,51,69,118 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,32,117,105,110,116,51 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 167                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 9152                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x23b3:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9147                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x23bb:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 8990                            // DW_AT_type
; CHECK-NEXT: .b8 20                               // Abbrev [20] 0x23c0:0xe DW_TAG_typedef
; CHECK-NEXT: .b32 8360                            // DW_AT_type
; CHECK-NEXT: .b8 117,105,110,116,51               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 14                               // DW_AT_decl_file
; CHECK-NEXT: .b8 127                              // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x23ce:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 9171                            // DW_AT_type
; CHECK-NEXT: .b8 9                                // Abbrev [9] 0x23d3:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 8438                            // DW_AT_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x23d8:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 8438                            // DW_AT_type
; CHECK-NEXT: .b8 33                               // Abbrev [33] 0x23dd:0x5 DW_TAG_reference_type
; CHECK-NEXT: .b32 9171                            // DW_AT_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x23e2:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 8438                            // DW_AT_type
; CHECK-NEXT: .b8 34                               // Abbrev [34] 0x23e7:0x6 DW_TAG_subprogram
; CHECK-NEXT: .b32 8468                            // DW_AT_specification
; CHECK-NEXT: .b8 1                                // DW_AT_inline
; CHECK-NEXT: .b8 27                               // Abbrev [27] 0x23ed:0x233 DW_TAG_structure_type
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_byte_size
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 66                               // DW_AT_decl_line
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x240c:0x50 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,54,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116,49,55,95,95,102,101,116,99,104 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 95,98,117,105,108,116,105,110,95,120,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,120 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 67                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x245c:0x50 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,54,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116,49,55,95,95,102,101,116,99,104 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 95,98,117,105,108,116,105,110,95,121,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,121 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 68                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 28                               // Abbrev [28] 0x24ac:0x50 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,50,54,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116,49,55,95,95,102,101,116,99,104 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 95,98,117,105,108,116,105,110,95,122,69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95,95,102,101,116,99,104,95,98,117,105,108,116,105,110,95,122 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 69                               // DW_AT_decl_line
; CHECK-NEXT: .b32 5207                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 25                               // Abbrev [25] 0x24fc:0x4a DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,54,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116,99,118,53,117,105,110,116,51 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 69,118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,32,117,105,110,116,51 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 72                               // DW_AT_decl_line
; CHECK-NEXT: .b32 8360                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x253f:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9760                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 30                               // Abbrev [30] 0x2546:0x28 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 74                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x2567:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9770                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 30                               // Abbrev [30] 0x256e:0x2d DW_TAG_subprogram
; CHECK-NEXT: .b8 95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 74                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x258f:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9770                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x2595:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9775                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 31                               // Abbrev [31] 0x259b:0x44 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,54,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116,97,83,69,82,75,83,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,61 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 74                               // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x25d3:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9760                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x25d9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9775                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 32                               // Abbrev [32] 0x25df:0x40 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,78,75,50,54,95,95,99,117,100,97,95,98,117,105,108,116,105,110,95,116,104,114,101,97,100,73,100,120,95,116,97,100,69,118 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111,112,101,114,97,116,111,114,38 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 13                               // DW_AT_decl_file
; CHECK-NEXT: .b8 74                               // DW_AT_decl_line
; CHECK-NEXT: .b32 9780                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // DW_AT_accessibility
; CHECK-NEXT:                                      // DW_ACCESS_private
; CHECK-NEXT: .b8 29                               // Abbrev [29] 0x2618:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9760                            // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_artificial
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x2620:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 9765                            // DW_AT_type
; CHECK-NEXT: .b8 9                                // Abbrev [9] 0x2625:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 9197                            // DW_AT_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x262a:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 9197                            // DW_AT_type
; CHECK-NEXT: .b8 33                               // Abbrev [33] 0x262f:0x5 DW_TAG_reference_type
; CHECK-NEXT: .b32 9765                            // DW_AT_type
; CHECK-NEXT: .b8 8                                // Abbrev [8] 0x2634:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 9197                            // DW_AT_type
; CHECK-NEXT: .b8 34                               // Abbrev [34] 0x2639:0x6 DW_TAG_subprogram
; CHECK-NEXT: .b32 9228                            // DW_AT_specification
; CHECK-NEXT: .b8 1                                // DW_AT_inline
; CHECK-NEXT: .b8 38                               // Abbrev [38] 0x263f:0x32 DW_TAG_subprogram
; CHECK-NEXT: .b8 95,90,51,114,101,115,102,102,80,102 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114,101,115                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 1                                // DW_AT_inline
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x2653:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 120                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x265c:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 121                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x2665:0xb DW_TAG_formal_parameter
; CHECK-NEXT: .b8 114,101,115                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b32 3345                            // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 40                               // Abbrev [40] 0x2671:0xc0 DW_TAG_subprogram
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 95,90,53,115,97,120,112,121,105,102,80,102,83,95 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115,97,120,112,121               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x269c:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 110                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x26a5:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 97                               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b32 1554                            // DW_AT_type
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x26ae:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 120                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b32 3345                            // DW_AT_type
; CHECK-NEXT: .b8 39                               // Abbrev [39] 0x26b7:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 121                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                // DW_AT_decl_line
; CHECK-NEXT: .b32 3345                            // DW_AT_type
; CHECK-NEXT: .b8 41                               // Abbrev [41] 0x26c0:0x9 DW_TAG_variable
; CHECK-NEXT: .b8 105                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                // DW_AT_decl_line
; CHECK-NEXT: .b32 2332                            // DW_AT_type
; CHECK-NEXT: .b8 42                               // Abbrev [42] 0x26c9:0x17 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 8432                            // DW_AT_abstract_origin
; CHECK-NEXT: .b64 Ltmp0                           // DW_AT_low_pc
; CHECK-NEXT: .b64 Ltmp1                           // DW_AT_high_pc
; CHECK-NEXT: .b8 12                               // DW_AT_call_file
; CHECK-NEXT: .b8 6                                // DW_AT_call_line
; CHECK-NEXT: .b8 42                               // Abbrev [42] 0x26e0:0x17 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 9191                            // DW_AT_abstract_origin
; CHECK-NEXT: .b64 Ltmp1                           // DW_AT_low_pc
; CHECK-NEXT: .b64 Ltmp2                           // DW_AT_high_pc
; CHECK-NEXT: .b8 12                               // DW_AT_call_file
; CHECK-NEXT: .b8 6                                // DW_AT_call_line
; CHECK-NEXT: .b8 42                               // Abbrev [42] 0x26f7:0x17 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 9785                            // DW_AT_abstract_origin
; CHECK-NEXT: .b64 Ltmp2                           // DW_AT_low_pc
; CHECK-NEXT: .b64 Ltmp3                           // DW_AT_high_pc
; CHECK-NEXT: .b8 12                               // DW_AT_call_file
; CHECK-NEXT: .b8 6                                // DW_AT_call_line
; CHECK-NEXT: .b8 43                               // Abbrev [43] 0x270e:0x22 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 9791                            // DW_AT_abstract_origin
; CHECK-NEXT: .b64 Ltmp10                          // DW_AT_low_pc
; CHECK-NEXT: .b64 Ltmp11                          // DW_AT_high_pc
; CHECK-NEXT: .b8 12                               // DW_AT_call_file
; CHECK-NEXT: .b8 8                                // DW_AT_call_line
; CHECK-NEXT: .b8 44                               // Abbrev [44] 0x2725:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9811                            // DW_AT_abstract_origin
; CHECK-NEXT: .b8 44                               // Abbrev [44] 0x272a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 9820                            // DW_AT_abstract_origin
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_macinfo
; CHECK-NEXT: {
; CHECK-NEXT: .b8 0                                // End Of Macro List Mark
; CHECK:      }

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!nvvm.annotations = !{!555, !556, !557, !556, !558, !558, !558, !558, !559, !559, !558}
!llvm.module.flags = !{!560, !561, !562, !563}
!llvm.ident = !{!564}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!565}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !3, nameTableKind: None)
!1 = !DIFile(filename: "debug-info.cu", directory: "/some/directory")
!2 = !{}
!3 = !{!4, !11, !16, !18, !20, !22, !24, !28, !30, !32, !34, !36, !38, !40, !42, !44, !46, !48, !50, !52, !54, !56, !60, !62, !64, !66, !71, !76, !78, !80, !85, !89, !91, !93, !95, !97, !99, !101, !103, !105, !110, !114, !116, !118, !122, !124, !126, !128, !130, !132, !136, !138, !140, !145, !153, !157, !159, !161, !163, !165, !169, !171, !173, !177, !179, !181, !183, !185, !187, !189, !191, !193, !195, !201, !203, !205, !209, !211, !213, !215, !217, !219, !221, !223, !227, !231, !233, !235, !240, !242, !244, !246, !248, !250, !252, !257, !263, !267, !271, !276, !279, !283, !287, !302, !306, !310, !314, !318, !323, !325, !329, !333, !337, !345, !349, !353, !357, !361, !366, !372, !376, !380, !382, !390, !394, !401, !403, !405, !409, !413, !417, !422, !426, !431, !432, !433, !434, !436, !437, !438, !439, !440, !441, !442, !446, !448, !450, !452, !454, !456, !458, !460, !463, !465, !467, !469, !471, !473, !475, !477, !479, !481, !483, !485, !487, !489, !491, !493, !495, !497, !499, !501, !503, !505, !507, !509, !511, !513, !515, !517, !519, !521, !523, !525, !527, !529, !531, !533, !535, !537, !539, !541, !543, !545, !547, !549, !551, !553}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !7, line: 202)
!5 = !DINamespace(name: "std", scope: null)
!6 = !DISubprogram(name: "abs", linkageName: "_ZL3absx", scope: !7, file: !7, line: 44, type: !8, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!7 = !DIFile(filename: "clang/include/__clang_cuda_math_forward_declares.h", directory: "/some/directory")
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!11 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !12, file: !7, line: 203)
!12 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !7, file: !7, line: 46, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15}
!15 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!16 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !17, file: !7, line: 204)
!17 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !7, file: !7, line: 48, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!18 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !19, file: !7, line: 205)
!19 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !7, file: !7, line: 50, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!20 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !21, file: !7, line: 206)
!21 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !7, file: !7, line: 52, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !23, file: !7, line: 207)
!23 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !7, file: !7, line: 56, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!24 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !25, file: !7, line: 208)
!25 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !7, file: !7, line: 54, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!26 = !DISubroutineType(types: !27)
!27 = !{!15, !15, !15}
!28 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !29, file: !7, line: 209)
!29 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !7, file: !7, line: 58, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!30 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !31, file: !7, line: 210)
!31 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !7, file: !7, line: 60, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!32 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !33, file: !7, line: 211)
!33 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !7, file: !7, line: 62, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !35, file: !7, line: 212)
!35 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !7, file: !7, line: 64, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !37, file: !7, line: 213)
!37 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !7, file: !7, line: 66, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !39, file: !7, line: 214)
!39 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !7, file: !7, line: 68, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!40 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !41, file: !7, line: 215)
!41 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !7, file: !7, line: 72, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !43, file: !7, line: 216)
!43 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !7, file: !7, line: 70, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !45, file: !7, line: 217)
!45 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !7, file: !7, line: 76, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!46 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !47, file: !7, line: 218)
!47 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !7, file: !7, line: 74, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!48 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !49, file: !7, line: 219)
!49 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !7, file: !7, line: 78, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !51, file: !7, line: 220)
!51 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !7, file: !7, line: 80, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!52 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !53, file: !7, line: 221)
!53 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !7, file: !7, line: 82, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !55, file: !7, line: 222)
!55 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !7, file: !7, line: 84, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!56 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !57, file: !7, line: 223)
!57 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !7, file: !7, line: 86, type: !58, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!58 = !DISubroutineType(types: !59)
!59 = !{!15, !15, !15, !15}
!60 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !61, file: !7, line: 224)
!61 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !7, file: !7, line: 88, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!62 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !63, file: !7, line: 225)
!63 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !7, file: !7, line: 90, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!64 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !65, file: !7, line: 226)
!65 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !7, file: !7, line: 92, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!66 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !67, file: !7, line: 227)
!67 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !7, file: !7, line: 94, type: !68, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!68 = !DISubroutineType(types: !69)
!69 = !{!70, !15}
!70 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !72, file: !7, line: 228)
!72 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !7, file: !7, line: 96, type: !73, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!73 = !DISubroutineType(types: !74)
!74 = !{!15, !15, !75}
!75 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !70, size: 64)
!76 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !77, file: !7, line: 229)
!77 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !7, file: !7, line: 98, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!78 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !79, file: !7, line: 230)
!79 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !7, file: !7, line: 100, type: !68, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!80 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !81, file: !7, line: 231)
!81 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !7, file: !7, line: 102, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!82 = !DISubroutineType(types: !83)
!83 = !{!84, !15}
!84 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!85 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !86, file: !7, line: 232)
!86 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !7, file: !7, line: 106, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!87 = !DISubroutineType(types: !88)
!88 = !{!84, !15, !15}
!89 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !90, file: !7, line: 233)
!90 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !7, file: !7, line: 105, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!91 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !92, file: !7, line: 234)
!92 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !7, file: !7, line: 108, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!93 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !94, file: !7, line: 235)
!94 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !7, file: !7, line: 112, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!95 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !96, file: !7, line: 236)
!96 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !7, file: !7, line: 111, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!97 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !98, file: !7, line: 237)
!98 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !7, file: !7, line: 114, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!99 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !100, file: !7, line: 238)
!100 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !7, file: !7, line: 116, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!101 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !102, file: !7, line: 239)
!102 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !7, file: !7, line: 118, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!103 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !104, file: !7, line: 240)
!104 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !7, file: !7, line: 120, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!105 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !106, file: !7, line: 241)
!106 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !7, file: !7, line: 121, type: !107, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!107 = !DISubroutineType(types: !108)
!108 = !{!109, !109}
!109 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!110 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !111, file: !7, line: 242)
!111 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !7, file: !7, line: 123, type: !112, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!112 = !DISubroutineType(types: !113)
!113 = !{!15, !15, !70}
!114 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !115, file: !7, line: 243)
!115 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !7, file: !7, line: 125, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!116 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !117, file: !7, line: 244)
!117 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !7, file: !7, line: 126, type: !8, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!118 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !119, file: !7, line: 245)
!119 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !7, file: !7, line: 128, type: !120, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!120 = !DISubroutineType(types: !121)
!121 = !{!10, !15}
!122 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !123, file: !7, line: 246)
!123 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !7, file: !7, line: 138, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !125, file: !7, line: 247)
!125 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !7, file: !7, line: 130, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !127, file: !7, line: 248)
!127 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !7, file: !7, line: 132, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!128 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !129, file: !7, line: 249)
!129 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !7, file: !7, line: 134, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !131, file: !7, line: 250)
!131 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !7, file: !7, line: 136, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!132 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !133, file: !7, line: 251)
!133 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !7, file: !7, line: 140, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!134 = !DISubroutineType(types: !135)
!135 = !{!109, !15}
!136 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !137, file: !7, line: 252)
!137 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !7, file: !7, line: 142, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !139, file: !7, line: 253)
!139 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !7, file: !7, line: 143, type: !120, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !141, file: !7, line: 254)
!141 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !7, file: !7, line: 145, type: !142, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!142 = !DISubroutineType(types: !143)
!143 = !{!15, !15, !144}
!144 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!145 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !146, file: !7, line: 255)
!146 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !7, file: !7, line: 146, type: !147, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!147 = !DISubroutineType(types: !148)
!148 = !{!149, !150}
!149 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!150 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !151, size: 64)
!151 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !152)
!152 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!153 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !154, file: !7, line: 256)
!154 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !7, file: !7, line: 147, type: !155, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!155 = !DISubroutineType(types: !156)
!156 = !{!15, !150}
!157 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !158, file: !7, line: 257)
!158 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !7, file: !7, line: 149, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !160, file: !7, line: 258)
!160 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !7, file: !7, line: 151, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!161 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !162, file: !7, line: 259)
!162 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !7, file: !7, line: 155, type: !112, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!163 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !164, file: !7, line: 260)
!164 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !7, file: !7, line: 157, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!165 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !166, file: !7, line: 261)
!166 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !7, file: !7, line: 159, type: !167, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!167 = !DISubroutineType(types: !168)
!168 = !{!15, !15, !15, !75}
!169 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !170, file: !7, line: 262)
!170 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !7, file: !7, line: 161, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !172, file: !7, line: 263)
!172 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !7, file: !7, line: 163, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !174, file: !7, line: 264)
!174 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !7, file: !7, line: 165, type: !175, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!175 = !DISubroutineType(types: !176)
!176 = !{!15, !15, !109}
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !178, file: !7, line: 265)
!178 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !7, file: !7, line: 167, type: !112, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !180, file: !7, line: 266)
!180 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !7, file: !7, line: 169, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !182, file: !7, line: 267)
!182 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !7, file: !7, line: 171, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !184, file: !7, line: 268)
!184 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !7, file: !7, line: 173, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !186, file: !7, line: 269)
!186 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !7, file: !7, line: 175, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !188, file: !7, line: 270)
!188 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !7, file: !7, line: 177, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !190, file: !7, line: 271)
!190 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !7, file: !7, line: 179, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !192, file: !7, line: 272)
!192 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !7, file: !7, line: 181, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !194, file: !7, line: 273)
!194 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !7, file: !7, line: 183, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !196, file: !200, line: 102)
!196 = !DISubprogram(name: "acos", scope: !197, file: !197, line: 54, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!197 = !DIFile(filename: "/usr/include/mathcalls.h", directory: "/some/directory")
!198 = !DISubroutineType(types: !199)
!199 = !{!149, !149}
!200 = !DIFile(filename: "/usr/lib/gcc/4.8/../../../../include/c++/4.8/cmath", directory: "/some/directory")
!201 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !202, file: !200, line: 121)
!202 = !DISubprogram(name: "asin", scope: !197, file: !197, line: 56, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !204, file: !200, line: 140)
!204 = !DISubprogram(name: "atan", scope: !197, file: !197, line: 58, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !206, file: !200, line: 159)
!206 = !DISubprogram(name: "atan2", scope: !197, file: !197, line: 60, type: !207, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!207 = !DISubroutineType(types: !208)
!208 = !{!149, !149, !149}
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !210, file: !200, line: 180)
!210 = !DISubprogram(name: "ceil", scope: !197, file: !197, line: 178, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !212, file: !200, line: 199)
!212 = !DISubprogram(name: "cos", scope: !197, file: !197, line: 63, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !214, file: !200, line: 218)
!214 = !DISubprogram(name: "cosh", scope: !197, file: !197, line: 72, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !216, file: !200, line: 237)
!216 = !DISubprogram(name: "exp", scope: !197, file: !197, line: 100, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !218, file: !200, line: 256)
!218 = !DISubprogram(name: "fabs", scope: !197, file: !197, line: 181, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!219 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !220, file: !200, line: 275)
!220 = !DISubprogram(name: "floor", scope: !197, file: !197, line: 184, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !222, file: !200, line: 294)
!222 = !DISubprogram(name: "fmod", scope: !197, file: !197, line: 187, type: !207, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!223 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !224, file: !200, line: 315)
!224 = !DISubprogram(name: "frexp", scope: !197, file: !197, line: 103, type: !225, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!225 = !DISubroutineType(types: !226)
!226 = !{!149, !149, !75}
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !228, file: !200, line: 334)
!228 = !DISubprogram(name: "ldexp", scope: !197, file: !197, line: 106, type: !229, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!229 = !DISubroutineType(types: !230)
!230 = !{!149, !149, !70}
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !232, file: !200, line: 353)
!232 = !DISubprogram(name: "log", scope: !197, file: !197, line: 109, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !234, file: !200, line: 372)
!234 = !DISubprogram(name: "log10", scope: !197, file: !197, line: 112, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !236, file: !200, line: 391)
!236 = !DISubprogram(name: "modf", scope: !197, file: !197, line: 115, type: !237, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!237 = !DISubroutineType(types: !238)
!238 = !{!149, !149, !239}
!239 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !149, size: 64)
!240 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !241, file: !200, line: 403)
!241 = !DISubprogram(name: "pow", scope: !197, file: !197, line: 153, type: !207, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!242 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !243, file: !200, line: 440)
!243 = !DISubprogram(name: "sin", scope: !197, file: !197, line: 65, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!244 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !245, file: !200, line: 459)
!245 = !DISubprogram(name: "sinh", scope: !197, file: !197, line: 74, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!246 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !247, file: !200, line: 478)
!247 = !DISubprogram(name: "sqrt", scope: !197, file: !197, line: 156, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!248 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !249, file: !200, line: 497)
!249 = !DISubprogram(name: "tan", scope: !197, file: !197, line: 67, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!250 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !251, file: !200, line: 516)
!251 = !DISubprogram(name: "tanh", scope: !197, file: !197, line: 76, type: !198, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !253, file: !256, line: 118)
!253 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !254, line: 101, baseType: !255)
!254 = !DIFile(filename: "/usr/include/stdlib.h", directory: "/some/directory")
!255 = !DICompositeType(tag: DW_TAG_structure_type, file: !254, line: 97, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!256 = !DIFile(filename: "/usr/lib/gcc/4.8/../../../../include/c++/4.8/cstdlib", directory: "/some/directory")
!257 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !258, file: !256, line: 119)
!258 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !254, line: 109, baseType: !259)
!259 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !254, line: 105, size: 128, elements: !260, identifier: "_ZTS6ldiv_t")
!260 = !{!261, !262}
!261 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !259, file: !254, line: 107, baseType: !109, size: 64)
!262 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !259, file: !254, line: 108, baseType: !109, size: 64, offset: 64)
!263 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !264, file: !256, line: 121)
!264 = !DISubprogram(name: "abort", scope: !254, file: !254, line: 515, type: !265, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: true)
!265 = !DISubroutineType(types: !266)
!266 = !{null}
!267 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !268, file: !256, line: 122)
!268 = !DISubprogram(name: "abs", scope: !254, file: !254, line: 775, type: !269, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!269 = !DISubroutineType(types: !270)
!270 = !{!70, !70}
!271 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !272, file: !256, line: 123)
!272 = !DISubprogram(name: "atexit", scope: !254, file: !254, line: 519, type: !273, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!273 = !DISubroutineType(types: !274)
!274 = !{!70, !275}
!275 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !265, size: 64)
!276 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !277, file: !256, line: 129)
!277 = !DISubprogram(name: "atof", scope: !278, file: !278, line: 26, type: !147, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!278 = !DIFile(filename: "/usr/include/stdlib-float.h", directory: "/some/directory")
!279 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !280, file: !256, line: 130)
!280 = !DISubprogram(name: "atoi", scope: !254, file: !254, line: 278, type: !281, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!281 = !DISubroutineType(types: !282)
!282 = !{!70, !150}
!283 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !284, file: !256, line: 131)
!284 = !DISubprogram(name: "atol", scope: !254, file: !254, line: 283, type: !285, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!285 = !DISubroutineType(types: !286)
!286 = !{!109, !150}
!287 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !288, file: !256, line: 132)
!288 = !DISubprogram(name: "bsearch", scope: !289, file: !289, line: 20, type: !290, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!289 = !DIFile(filename: "/usr/include/stdlib-bsearch.h", directory: "/some/directory")
!290 = !DISubroutineType(types: !291)
!291 = !{!292, !293, !293, !295, !295, !298}
!292 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!293 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !294, size: 64)
!294 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!295 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !296, line: 62, baseType: !297)
!296 = !DIFile(filename: "clang/include/stddef.h", directory: "/some/directory")
!297 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!298 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !254, line: 742, baseType: !299)
!299 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !300, size: 64)
!300 = !DISubroutineType(types: !301)
!301 = !{!70, !293, !293}
!302 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !303, file: !256, line: 133)
!303 = !DISubprogram(name: "calloc", scope: !254, file: !254, line: 468, type: !304, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!304 = !DISubroutineType(types: !305)
!305 = !{!292, !295, !295}
!306 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !307, file: !256, line: 134)
!307 = !DISubprogram(name: "div", scope: !254, file: !254, line: 789, type: !308, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!308 = !DISubroutineType(types: !309)
!309 = !{!253, !70, !70}
!310 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !311, file: !256, line: 135)
!311 = !DISubprogram(name: "exit", scope: !254, file: !254, line: 543, type: !312, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: true)
!312 = !DISubroutineType(types: !313)
!313 = !{null, !70}
!314 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !315, file: !256, line: 136)
!315 = !DISubprogram(name: "free", scope: !254, file: !254, line: 483, type: !316, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!316 = !DISubroutineType(types: !317)
!317 = !{null, !292}
!318 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !319, file: !256, line: 137)
!319 = !DISubprogram(name: "getenv", scope: !254, file: !254, line: 564, type: !320, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!320 = !DISubroutineType(types: !321)
!321 = !{!322, !150}
!322 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !152, size: 64)
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !324, file: !256, line: 138)
!324 = !DISubprogram(name: "labs", scope: !254, file: !254, line: 776, type: !107, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!325 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !326, file: !256, line: 139)
!326 = !DISubprogram(name: "ldiv", scope: !254, file: !254, line: 791, type: !327, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!327 = !DISubroutineType(types: !328)
!328 = !{!258, !109, !109}
!329 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !330, file: !256, line: 140)
!330 = !DISubprogram(name: "malloc", scope: !254, file: !254, line: 466, type: !331, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!331 = !DISubroutineType(types: !332)
!332 = !{!292, !295}
!333 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !334, file: !256, line: 142)
!334 = !DISubprogram(name: "mblen", scope: !254, file: !254, line: 863, type: !335, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!335 = !DISubroutineType(types: !336)
!336 = !{!70, !150, !295}
!337 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !338, file: !256, line: 143)
!338 = !DISubprogram(name: "mbstowcs", scope: !254, file: !254, line: 874, type: !339, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!339 = !DISubroutineType(types: !340)
!340 = !{!295, !341, !344, !295}
!341 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !342)
!342 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !343, size: 64)
!343 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!344 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !150)
!345 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !346, file: !256, line: 144)
!346 = !DISubprogram(name: "mbtowc", scope: !254, file: !254, line: 866, type: !347, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!347 = !DISubroutineType(types: !348)
!348 = !{!70, !341, !344, !295}
!349 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !350, file: !256, line: 146)
!350 = !DISubprogram(name: "qsort", scope: !254, file: !254, line: 765, type: !351, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!351 = !DISubroutineType(types: !352)
!352 = !{null, !292, !295, !295, !298}
!353 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !354, file: !256, line: 152)
!354 = !DISubprogram(name: "rand", scope: !254, file: !254, line: 374, type: !355, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!355 = !DISubroutineType(types: !356)
!356 = !{!70}
!357 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !358, file: !256, line: 153)
!358 = !DISubprogram(name: "realloc", scope: !254, file: !254, line: 480, type: !359, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!359 = !DISubroutineType(types: !360)
!360 = !{!292, !292, !295}
!361 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !362, file: !256, line: 154)
!362 = !DISubprogram(name: "srand", scope: !254, file: !254, line: 376, type: !363, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!363 = !DISubroutineType(types: !364)
!364 = !{null, !365}
!365 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !367, file: !256, line: 155)
!367 = !DISubprogram(name: "strtod", scope: !254, file: !254, line: 164, type: !368, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!368 = !DISubroutineType(types: !369)
!369 = !{!149, !344, !370}
!370 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !371)
!371 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !322, size: 64)
!372 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !373, file: !256, line: 156)
!373 = !DISubprogram(name: "strtol", scope: !254, file: !254, line: 183, type: !374, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!374 = !DISubroutineType(types: !375)
!375 = !{!109, !344, !370, !70}
!376 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !377, file: !256, line: 157)
!377 = !DISubprogram(name: "strtoul", scope: !254, file: !254, line: 187, type: !378, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!378 = !DISubroutineType(types: !379)
!379 = !{!297, !344, !370, !70}
!380 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !381, file: !256, line: 158)
!381 = !DISubprogram(name: "system", scope: !254, file: !254, line: 717, type: !281, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!382 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !383, file: !256, line: 160)
!383 = !DISubprogram(name: "wcstombs", scope: !254, file: !254, line: 877, type: !384, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!384 = !DISubroutineType(types: !385)
!385 = !{!295, !386, !387, !295}
!386 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !322)
!387 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !388)
!388 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !389, size: 64)
!389 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !343)
!390 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !391, file: !256, line: 161)
!391 = !DISubprogram(name: "wctomb", scope: !254, file: !254, line: 870, type: !392, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!392 = !DISubroutineType(types: !393)
!393 = !{!70, !322, !343}
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !396, file: !256, line: 201)
!395 = !DINamespace(name: "__gnu_cxx", scope: null)
!396 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !254, line: 121, baseType: !397)
!397 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !254, line: 117, size: 128, elements: !398, identifier: "_ZTS7lldiv_t")
!398 = !{!399, !400}
!399 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !397, file: !254, line: 119, baseType: !10, size: 64)
!400 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !397, file: !254, line: 120, baseType: !10, size: 64, offset: 64)
!401 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !402, file: !256, line: 207)
!402 = !DISubprogram(name: "_Exit", scope: !254, file: !254, line: 557, type: !312, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: true)
!403 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !404, file: !256, line: 211)
!404 = !DISubprogram(name: "llabs", scope: !254, file: !254, line: 780, type: !8, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!405 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !406, file: !256, line: 217)
!406 = !DISubprogram(name: "lldiv", scope: !254, file: !254, line: 797, type: !407, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!407 = !DISubroutineType(types: !408)
!408 = !{!396, !10, !10}
!409 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !410, file: !256, line: 228)
!410 = !DISubprogram(name: "atoll", scope: !254, file: !254, line: 292, type: !411, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!411 = !DISubroutineType(types: !412)
!412 = !{!10, !150}
!413 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !414, file: !256, line: 229)
!414 = !DISubprogram(name: "strtoll", scope: !254, file: !254, line: 209, type: !415, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!415 = !DISubroutineType(types: !416)
!416 = !{!10, !344, !370, !70}
!417 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !418, file: !256, line: 230)
!418 = !DISubprogram(name: "strtoull", scope: !254, file: !254, line: 214, type: !419, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!419 = !DISubroutineType(types: !420)
!420 = !{!421, !344, !370, !70}
!421 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !423, file: !256, line: 232)
!423 = !DISubprogram(name: "strtof", scope: !254, file: !254, line: 172, type: !424, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!424 = !DISubroutineType(types: !425)
!425 = !{!15, !344, !370}
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !395, entity: !427, file: !256, line: 233)
!427 = !DISubprogram(name: "strtold", scope: !254, file: !254, line: 175, type: !428, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!428 = !DISubroutineType(types: !429)
!429 = !{!430, !344, !370}
!430 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!431 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !396, file: !256, line: 241)
!432 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !402, file: !256, line: 243)
!433 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !404, file: !256, line: 245)
!434 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !435, file: !256, line: 246)
!435 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !395, file: !256, line: 214, type: !407, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !406, file: !256, line: 247)
!437 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !410, file: !256, line: 249)
!438 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !423, file: !256, line: 250)
!439 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !414, file: !256, line: 251)
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !418, file: !256, line: 252)
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !427, file: !256, line: 253)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !443, file: !445, line: 405)
!443 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !444, file: !444, line: 1342, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!444 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "/some/directory")
!445 = !DIFile(filename: "clang/include/__clang_cuda_cmath.h", directory: "/some/directory")
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !447, file: !445, line: 406)
!447 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !444, file: !444, line: 1370, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !449, file: !445, line: 407)
!449 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !444, file: !444, line: 1337, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !451, file: !445, line: 408)
!451 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !444, file: !444, line: 1375, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !453, file: !445, line: 409)
!453 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !444, file: !444, line: 1327, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !455, file: !445, line: 410)
!455 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !444, file: !444, line: 1332, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !457, file: !445, line: 411)
!457 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !444, file: !444, line: 1380, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !459, file: !445, line: 412)
!459 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !444, file: !444, line: 1430, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !461, file: !445, line: 413)
!461 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !462, file: !462, line: 667, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!462 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "/some/directory")
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !464, file: !445, line: 414)
!464 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !444, file: !444, line: 1189, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !466, file: !445, line: 415)
!466 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !444, file: !444, line: 1243, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !468, file: !445, line: 416)
!468 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !444, file: !444, line: 1312, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !470, file: !445, line: 417)
!470 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !444, file: !444, line: 1490, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !472, file: !445, line: 418)
!472 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !444, file: !444, line: 1480, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !474, file: !445, line: 419)
!474 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !462, file: !462, line: 657, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!475 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !476, file: !445, line: 420)
!476 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !444, file: !444, line: 1294, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !478, file: !445, line: 421)
!478 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !444, file: !444, line: 1385, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !480, file: !445, line: 422)
!480 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !462, file: !462, line: 607, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !482, file: !445, line: 423)
!482 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !444, file: !444, line: 1616, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!483 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !484, file: !445, line: 424)
!484 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !462, file: !462, line: 597, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !486, file: !445, line: 425)
!486 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !444, file: !444, line: 1568, type: !58, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!487 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !488, file: !445, line: 426)
!488 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !462, file: !462, line: 622, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !490, file: !445, line: 427)
!490 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !462, file: !462, line: 617, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!491 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !492, file: !445, line: 428)
!492 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !444, file: !444, line: 1553, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!493 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !494, file: !445, line: 429)
!494 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !444, file: !444, line: 1543, type: !73, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!495 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !496, file: !445, line: 430)
!496 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !444, file: !444, line: 1390, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!497 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !498, file: !445, line: 431)
!498 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !444, file: !444, line: 1621, type: !68, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !500, file: !445, line: 432)
!500 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !444, file: !444, line: 1520, type: !112, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!501 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !502, file: !445, line: 433)
!502 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !444, file: !444, line: 1515, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !504, file: !445, line: 434)
!504 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !444, file: !444, line: 1149, type: !120, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!505 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !506, file: !445, line: 435)
!506 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !444, file: !444, line: 1602, type: !120, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !508, file: !445, line: 436)
!508 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !444, file: !444, line: 1356, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!509 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !510, file: !445, line: 437)
!510 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !444, file: !444, line: 1365, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!511 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !512, file: !445, line: 438)
!512 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !444, file: !444, line: 1285, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!513 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !514, file: !445, line: 439)
!514 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !444, file: !444, line: 1626, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !516, file: !445, line: 440)
!516 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !444, file: !444, line: 1347, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!517 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !518, file: !445, line: 441)
!518 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !444, file: !444, line: 1140, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !520, file: !445, line: 442)
!520 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !444, file: !444, line: 1607, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!521 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !522, file: !445, line: 443)
!522 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !444, file: !444, line: 1548, type: !142, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!523 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !524, file: !445, line: 444)
!524 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !444, file: !444, line: 1154, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !526, file: !445, line: 445)
!526 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !444, file: !444, line: 1218, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!527 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !528, file: !445, line: 446)
!528 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !444, file: !444, line: 1583, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !530, file: !445, line: 447)
!530 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !444, file: !444, line: 1558, type: !26, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!531 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !532, file: !445, line: 448)
!532 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !444, file: !444, line: 1563, type: !167, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !534, file: !445, line: 449)
!534 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !444, file: !444, line: 1135, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!535 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !536, file: !445, line: 450)
!536 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !444, file: !444, line: 1597, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!537 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !538, file: !445, line: 451)
!538 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !444, file: !444, line: 1530, type: !175, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!539 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !540, file: !445, line: 452)
!540 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !444, file: !444, line: 1525, type: !112, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !542, file: !445, line: 453)
!542 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !444, file: !444, line: 1234, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!543 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !544, file: !445, line: 454)
!544 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !444, file: !444, line: 1317, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !546, file: !445, line: 455)
!546 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !462, file: !462, line: 907, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !548, file: !445, line: 456)
!548 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !444, file: !444, line: 1276, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !550, file: !445, line: 457)
!550 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !444, file: !444, line: 1322, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!551 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !552, file: !445, line: 458)
!552 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !444, file: !444, line: 1592, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !554, file: !445, line: 459)
!554 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !462, file: !462, line: 662, type: !13, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!555 = !{void (i32, float, float*, float*)* @_Z5saxpyifPfS_, !"kernel", i32 1}
!556 = !{null, !"align", i32 8}
!557 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!558 = !{null, !"align", i32 16}
!559 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!560 = !{i32 2, !"Dwarf Version", i32 2}
!561 = !{i32 2, !"Debug Info Version", i32 3}
!562 = !{i32 1, !"wchar_size", i32 4}
!563 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!564 = !{!""}
!565 = !{i32 1, i32 2}
!566 = distinct !DISubprogram(name: "saxpy", linkageName: "_Z5saxpyifPfS_", scope: !1, file: !1, line: 5, type: !567, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !569)
!567 = !DISubroutineType(types: !568)
!568 = !{null, !70, !15, !144, !144}
!569 = !{!570, !571, !572, !573, !574}
!570 = !DILocalVariable(name: "n", arg: 1, scope: !566, file: !1, line: 5, type: !70)
!571 = !DILocalVariable(name: "a", arg: 2, scope: !566, file: !1, line: 5, type: !15)
!572 = !DILocalVariable(name: "x", arg: 3, scope: !566, file: !1, line: 5, type: !144)
!573 = !DILocalVariable(name: "y", arg: 4, scope: !566, file: !1, line: 5, type: !144)
!574 = !DILocalVariable(name: "i", scope: !566, file: !1, line: 6, type: !70)
!575 = !DILocation(line: 5, column: 40, scope: !566)
!576 = !DILocation(line: 5, column: 49, scope: !566)
!577 = !DILocation(line: 5, column: 59, scope: !566)
!578 = !DILocation(line: 5, column: 69, scope: !566)
!579 = !DILocation(line: 78, column: 180, scope: !580, inlinedAt: !615)
!580 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !582, file: !581, line: 78, type: !585, isLocal: false, isDefinition: true, scopeLine: 78, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !584, retainedNodes: !2)
!581 = !DIFile(filename: "clang/include/__clang_cuda_builtin_vars.h", directory: "/some/directory")
!582 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !581, line: 77, size: 8, elements: !583, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!583 = !{!584, !587, !588, !589, !600, !604, !608, !611}
!584 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !582, file: !581, line: 78, type: !585, isLocal: false, isDefinition: false, scopeLine: 78, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!585 = !DISubroutineType(types: !586)
!586 = !{!365}
!587 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !582, file: !581, line: 79, type: !585, isLocal: false, isDefinition: false, scopeLine: 79, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!588 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !582, file: !581, line: 80, type: !585, isLocal: false, isDefinition: false, scopeLine: 80, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!589 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !582, file: !581, line: 83, type: !590, isLocal: false, isDefinition: false, scopeLine: 83, flags: DIFlagPrototyped, isOptimized: true)
!590 = !DISubroutineType(types: !591)
!591 = !{!592, !598}
!592 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !593, line: 190, size: 96, elements: !594, identifier: "_ZTS5uint3")
!593 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "/some/directory")
!594 = !{!595, !596, !597}
!595 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !592, file: !593, line: 192, baseType: !365, size: 32)
!596 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !592, file: !593, line: 192, baseType: !365, size: 32, offset: 32)
!597 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !592, file: !593, line: 192, baseType: !365, size: 32, offset: 64)
!598 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !599, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!599 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !582)
!600 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !582, file: !581, line: 85, type: !601, isLocal: false, isDefinition: false, scopeLine: 85, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!601 = !DISubroutineType(types: !602)
!602 = !{null, !603}
!603 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !582, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!604 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !582, file: !581, line: 85, type: !605, isLocal: false, isDefinition: false, scopeLine: 85, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!605 = !DISubroutineType(types: !606)
!606 = !{null, !603, !607}
!607 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !599, size: 64)
!608 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !582, file: !581, line: 85, type: !609, isLocal: false, isDefinition: false, scopeLine: 85, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!609 = !DISubroutineType(types: !610)
!610 = !{null, !598, !607}
!611 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !582, file: !581, line: 85, type: !612, isLocal: false, isDefinition: false, scopeLine: 85, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!612 = !DISubroutineType(types: !613)
!613 = !{!614, !598}
!614 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !582, size: 64)
!615 = distinct !DILocation(line: 6, column: 11, scope: !566)
!616 = !{i32 0, i32 65535}
!617 = !DILocation(line: 89, column: 180, scope: !618, inlinedAt: !660)
!618 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !619, file: !581, line: 89, type: !585, isLocal: false, isDefinition: true, scopeLine: 89, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !621, retainedNodes: !2)
!619 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !581, line: 88, size: 8, elements: !620, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!620 = !{!621, !622, !623, !624, !645, !649, !653, !656}
!621 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !619, file: !581, line: 89, type: !585, isLocal: false, isDefinition: false, scopeLine: 89, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!622 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !619, file: !581, line: 90, type: !585, isLocal: false, isDefinition: false, scopeLine: 90, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!623 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !619, file: !581, line: 91, type: !585, isLocal: false, isDefinition: false, scopeLine: 91, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!624 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !619, file: !581, line: 94, type: !625, isLocal: false, isDefinition: false, scopeLine: 94, flags: DIFlagPrototyped, isOptimized: true)
!625 = !DISubroutineType(types: !626)
!626 = !{!627, !643}
!627 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !593, line: 417, size: 96, elements: !628, identifier: "_ZTS4dim3")
!628 = !{!629, !630, !631, !632, !636, !640}
!629 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !627, file: !593, line: 419, baseType: !365, size: 32)
!630 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !627, file: !593, line: 419, baseType: !365, size: 32, offset: 32)
!631 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !627, file: !593, line: 419, baseType: !365, size: 32, offset: 64)
!632 = !DISubprogram(name: "dim3", scope: !627, file: !593, line: 421, type: !633, isLocal: false, isDefinition: false, scopeLine: 421, flags: DIFlagPrototyped, isOptimized: true)
!633 = !DISubroutineType(types: !634)
!634 = !{null, !635, !365, !365, !365}
!635 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !627, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!636 = !DISubprogram(name: "dim3", scope: !627, file: !593, line: 422, type: !637, isLocal: false, isDefinition: false, scopeLine: 422, flags: DIFlagPrototyped, isOptimized: true)
!637 = !DISubroutineType(types: !638)
!638 = !{null, !635, !639}
!639 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !593, line: 383, baseType: !592)
!640 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !627, file: !593, line: 423, type: !641, isLocal: false, isDefinition: false, scopeLine: 423, flags: DIFlagPrototyped, isOptimized: true)
!641 = !DISubroutineType(types: !642)
!642 = !{!639, !635}
!643 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !644, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!644 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !619)
!645 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !619, file: !581, line: 96, type: !646, isLocal: false, isDefinition: false, scopeLine: 96, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!646 = !DISubroutineType(types: !647)
!647 = !{null, !648}
!648 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !619, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!649 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !619, file: !581, line: 96, type: !650, isLocal: false, isDefinition: false, scopeLine: 96, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!650 = !DISubroutineType(types: !651)
!651 = !{null, !648, !652}
!652 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !644, size: 64)
!653 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !619, file: !581, line: 96, type: !654, isLocal: false, isDefinition: false, scopeLine: 96, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!654 = !DISubroutineType(types: !655)
!655 = !{null, !643, !652}
!656 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !619, file: !581, line: 96, type: !657, isLocal: false, isDefinition: false, scopeLine: 96, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!657 = !DISubroutineType(types: !658)
!658 = !{!659, !643}
!659 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !619, size: 64)
!660 = distinct !DILocation(line: 6, column: 24, scope: !566)
!661 = !{i32 1, i32 1025}
!662 = !DILocation(line: 6, column: 22, scope: !566)
!663 = !DILocation(line: 67, column: 180, scope: !664, inlinedAt: !690)
!664 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !665, file: !581, line: 67, type: !585, isLocal: false, isDefinition: true, scopeLine: 67, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !667, retainedNodes: !2)
!665 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !581, line: 66, size: 8, elements: !666, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!666 = !{!667, !668, !669, !670, !675, !679, !683, !686}
!667 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !665, file: !581, line: 67, type: !585, isLocal: false, isDefinition: false, scopeLine: 67, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!668 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !665, file: !581, line: 68, type: !585, isLocal: false, isDefinition: false, scopeLine: 68, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!669 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !665, file: !581, line: 69, type: !585, isLocal: false, isDefinition: false, scopeLine: 69, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: true)
!670 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !665, file: !581, line: 72, type: !671, isLocal: false, isDefinition: false, scopeLine: 72, flags: DIFlagPrototyped, isOptimized: true)
!671 = !DISubroutineType(types: !672)
!672 = !{!592, !673}
!673 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !674, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!674 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !665)
!675 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !665, file: !581, line: 74, type: !676, isLocal: false, isDefinition: false, scopeLine: 74, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!676 = !DISubroutineType(types: !677)
!677 = !{null, !678}
!678 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !665, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!679 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !665, file: !581, line: 74, type: !680, isLocal: false, isDefinition: false, scopeLine: 74, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!680 = !DISubroutineType(types: !681)
!681 = !{null, !678, !682}
!682 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !674, size: 64)
!683 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !665, file: !581, line: 74, type: !684, isLocal: false, isDefinition: false, scopeLine: 74, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!684 = !DISubroutineType(types: !685)
!685 = !{null, !673, !682}
!686 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !665, file: !581, line: 74, type: !687, isLocal: false, isDefinition: false, scopeLine: 74, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: true)
!687 = !DISubroutineType(types: !688)
!688 = !{!689, !673}
!689 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !665, size: 64)
!690 = distinct !DILocation(line: 6, column: 37, scope: !566)
!691 = !{i32 0, i32 1024}
!692 = !DILocation(line: 6, column: 35, scope: !566)
!693 = !DILocation(line: 6, column: 7, scope: !566)
!694 = !DILocation(line: 7, column: 9, scope: !695)
!695 = distinct !DILexicalBlock(scope: !566, file: !1, line: 7, column: 7)
!696 = !DILocation(line: 7, column: 7, scope: !566)
!697 = !DILocation(line: 8, column: 13, scope: !695)
!698 = !{!699, !699, i64 0}
!699 = !{!"float", !700, i64 0}
!700 = !{!"omnipotent char", !701, i64 0}
!701 = !{!"Simple C++ TBAA"}
!702 = !DILocation(line: 8, column: 11, scope: !695)
!703 = !DILocation(line: 8, column: 19, scope: !695)
!704 = !DILocalVariable(name: "x", arg: 1, scope: !705, file: !1, line: 3, type: !15)
!705 = distinct !DISubprogram(name: "res", linkageName: "_Z3resffPf", scope: !1, file: !1, line: 3, type: !706, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !708)
!706 = !DISubroutineType(types: !707)
!707 = !{null, !15, !15, !144}
!708 = !{!704, !709, !710}
!709 = !DILocalVariable(name: "y", arg: 2, scope: !705, file: !1, line: 3, type: !15)
!710 = !DILocalVariable(name: "res", arg: 3, scope: !705, file: !1, line: 3, type: !144)
!711 = !DILocation(line: 3, column: 47, scope: !705, inlinedAt: !712)
!712 = distinct !DILocation(line: 8, column: 5, scope: !695)
!713 = !DILocation(line: 3, column: 56, scope: !705, inlinedAt: !712)
!714 = !DILocation(line: 3, column: 66, scope: !705, inlinedAt: !712)
!715 = !DILocation(line: 3, column: 82, scope: !705, inlinedAt: !712)
!716 = !DILocation(line: 3, column: 78, scope: !705, inlinedAt: !712)
!717 = !DILocation(line: 8, column: 5, scope: !695)
!718 = !DILocation(line: 9, column: 1, scope: !566)
