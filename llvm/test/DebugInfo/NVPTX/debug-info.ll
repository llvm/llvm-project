; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mattr=+ptx70 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -mattr=+ptx70 | %ptxas-verify %}

; // Bitcode in this test case is reduced version of compiled code below:
;__device__ inline void res(float x, float y, ptr res) { *res = x + y; }
;
;__global__ void saxpy(int n, float a, ptr x, ptr y) {
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
; CHECK-DAG: .reg .pred      %p<2>;
; CHECK-DAG: .reg .f32       %f<5>;
; CHECK-DAG: .reg .b32       %r<6>;
; CHECK-DAG: .reg .b64       %rd<8>;
; CHECK: .loc [[DEBUG_INFO_CU:[0-9]+]] 5 0
; CHECK: ld.param.u32    %r{{.+}}, [{{.+}}];
; CHECK: ld.param.u64    %rd{{.+}}, [{{.+}}];
; CHECK: cvta.to.global.u64      %rd{{.+}}, %rd{{.+}};
; CHECK: ld.param.u64    %rd{{.+}}, [{{.+}}];
; CHECK: cvta.to.global.u64      %rd{{.+}}, %rd{{.+}};
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
; CHECK: @%p{{.+}} bra   [[BB:\$L__.+]];
; CHECK: ld.param.f32    %f{{.+}}, [{{.+}}];
; CHECK: .loc [[DEBUG_INFO_CU]] 8 13
; CHECK: mul.wide.u32    %rd{{.+}}, %r{{.+}}, 4;
; CHECK: add.s64         %rd{{.+}}, %rd{{.+}}, %rd{{.+}};
; CHECK: ld.global.f32   %f{{.+}}, [%rd{{.+}}];
; CHECK: .loc [[DEBUG_INFO_CU]] 8 19
; CHECK: add.s64         %rd{{.+}}, %rd{{.+}}, %rd{{.+}};
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
define ptx_kernel void @_Z5saxpyifPfS_(i32 %n, float %a, ptr nocapture readonly %x, ptr nocapture %y) local_unnamed_addr #0 !dbg !566 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !570, metadata !DIExpression()), !dbg !575
  call void @llvm.dbg.value(metadata float %a, metadata !571, metadata !DIExpression()), !dbg !576
  call void @llvm.dbg.value(metadata ptr %x, metadata !572, metadata !DIExpression()), !dbg !577
  call void @llvm.dbg.value(metadata ptr %y, metadata !573, metadata !DIExpression()), !dbg !578
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
  %arrayidx = getelementptr inbounds float, ptr %x, i64 %3, !dbg !697
  %4 = load float, ptr %arrayidx, align 4, !dbg !697, !tbaa !698
  %mul3 = fmul contract float %4, %a, !dbg !702
  %arrayidx5 = getelementptr inbounds float, ptr %y, i64 %3, !dbg !703
  %5 = load float, ptr %arrayidx5, align 4, !dbg !703, !tbaa !698
  call void @llvm.dbg.value(metadata float %mul3, metadata !704, metadata !DIExpression()), !dbg !711
  call void @llvm.dbg.value(metadata float %5, metadata !709, metadata !DIExpression()), !dbg !713
  call void @llvm.dbg.value(metadata ptr %arrayidx5, metadata !710, metadata !DIExpression()), !dbg !714
  %add.i = fadd contract float %mul3, %5, !dbg !715
  store float %add.i, ptr %arrayidx5, align 4, !dbg !716, !tbaa !698
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
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}clang/include{{/|\\\\}}__clang_cuda_cmath.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/local/cuda/include{{/|\\\\}}device_functions.hpp"
; CHECK-DAG: .file [[DEBUG_INFO_CU]] "{{.*}}debug-info.cu"
; CHECK-DAG: .file [[BUILTUIN_VARS_H]] "{{.*}}clang/include{{/|\\\\}}__clang_cuda_builtin_vars.h"

; CHECK:	.section	.debug_loc
; CHECK-NEXT:	{
; CHECK-NEXT:$L__debug_loc0:
; CHECK-NEXT:.b64 $L__tmp8
; CHECK-NEXT:.b64 $L__tmp10
; CHECK-NEXT:.b8 5                                   // Loc expr size
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 144                                 // DW_OP_regx
; CHECK-NEXT:.b8 177                                 // 2450993
; CHECK-NEXT:.b8 204                                 // 
; CHECK-NEXT:.b8 149                                 // 
; CHECK-NEXT:.b8 1                                   // 
; CHECK-NEXT:.b64 0
; CHECK-NEXT:.b64 0
; CHECK-NEXT:$L__debug_loc1:
; CHECK-NEXT:.b64 $L__tmp5
; CHECK-NEXT:.b64 $L__func_end0
; CHECK-NEXT:.b8 5                                   // Loc expr size
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 144                                 // DW_OP_regx
; CHECK-NEXT:.b8 177                                 // 2454065
; CHECK-NEXT:.b8 228                                 // 
; CHECK-NEXT:.b8 149                                 // 
; CHECK-NEXT:.b8 1                                   // 
; CHECK-NEXT:.b64 0
; CHECK-NEXT:.b64 0
; CHECK-NEXT:	}
; CHECK-NEXT:	.section	.debug_abbrev
; CHECK-NEXT:	{
; CHECK-NEXT:.b8 1                                   // Abbreviation Code
; CHECK-NEXT:.b8 17                                  // DW_TAG_compile_unit
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 37                                  // DW_AT_producer
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 19                                  // DW_AT_language
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 16                                  // DW_AT_stmt_list
; CHECK-NEXT:.b8 6                                   // DW_FORM_data4
; CHECK-NEXT:.b8 27                                  // DW_AT_comp_dir
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 2                                   // Abbreviation Code
; CHECK-NEXT:.b8 19                                  // DW_TAG_structure_type
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 11                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 3                                   // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 4                                   // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 5                                   // Abbreviation Code
; CHECK-NEXT:.b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 52                                  // DW_AT_artificial
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 6                                   // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 50                                  // DW_AT_accessibility
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 7                                   // Abbreviation Code
; CHECK-NEXT:.b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 8                                   // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 50                                  // DW_AT_accessibility
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 9                                   // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 50                                  // DW_AT_accessibility
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 10                                  // Abbreviation Code
; CHECK-NEXT:.b8 36                                  // DW_TAG_base_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 62                                  // DW_AT_encoding
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 11                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 11                                  // Abbreviation Code
; CHECK-NEXT:.b8 13                                  // DW_TAG_member
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 56                                  // DW_AT_data_member_location
; CHECK-NEXT:.b8 10                                  // DW_FORM_block1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 12                                  // Abbreviation Code
; CHECK-NEXT:.b8 15                                  // DW_TAG_pointer_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 13                                  // Abbreviation Code
; CHECK-NEXT:.b8 38                                  // DW_TAG_const_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 14                                  // Abbreviation Code
; CHECK-NEXT:.b8 16                                  // DW_TAG_reference_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 15                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 71                                  // DW_AT_specification
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 32                                  // DW_AT_inline
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 16                                  // Abbreviation Code
; CHECK-NEXT:.b8 19                                  // DW_TAG_structure_type
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 11                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 17                                  // Abbreviation Code
; CHECK-NEXT:.b8 13                                  // DW_TAG_member
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 56                                  // DW_AT_data_member_location
; CHECK-NEXT:.b8 10                                  // DW_FORM_block1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 18                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 19                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 20                                  // Abbreviation Code
; CHECK-NEXT:.b8 22                                  // DW_TAG_typedef
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 21                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 32                                  // DW_AT_inline
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 22                                  // Abbreviation Code
; CHECK-NEXT:.b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 23                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 17                                  // DW_AT_low_pc
; CHECK-NEXT:.b8 1                                   // DW_FORM_addr
; CHECK-NEXT:.b8 18                                  // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_FORM_addr
; CHECK-NEXT:.b8 64                                  // DW_AT_frame_base
; CHECK-NEXT:.b8 10                                  // DW_FORM_block1
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 24                                  // Abbreviation Code
; CHECK-NEXT:.b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 51                                  // DW_AT_address_class
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 2                                   // DW_AT_location
; CHECK-NEXT:.b8 10                                  // DW_FORM_block1
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 25                                  // Abbreviation Code
; CHECK-NEXT:.b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 2                                   // DW_AT_location
; CHECK-NEXT:.b8 6                                   // DW_FORM_data4
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 26                                  // Abbreviation Code
; CHECK-NEXT:.b8 52                                  // DW_TAG_variable
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 2                                   // DW_AT_location
; CHECK-NEXT:.b8 6                                   // DW_FORM_data4
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 27                                  // Abbreviation Code
; CHECK-NEXT:.b8 29                                  // DW_TAG_inlined_subroutine
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 49                                  // DW_AT_abstract_origin
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 17                                  // DW_AT_low_pc
; CHECK-NEXT:.b8 1                                   // DW_FORM_addr
; CHECK-NEXT:.b8 18                                  // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_FORM_addr
; CHECK-NEXT:.b8 88                                  // DW_AT_call_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 89                                  // DW_AT_call_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 87                                  // DW_AT_call_column
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 28                                  // Abbreviation Code
; CHECK-NEXT:.b8 29                                  // DW_TAG_inlined_subroutine
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 49                                  // DW_AT_abstract_origin
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 17                                  // DW_AT_low_pc
; CHECK-NEXT:.b8 1                                   // DW_FORM_addr
; CHECK-NEXT:.b8 18                                  // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_FORM_addr
; CHECK-NEXT:.b8 88                                  // DW_AT_call_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 89                                  // DW_AT_call_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 87                                  // DW_AT_call_column
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 29                                  // Abbreviation Code
; CHECK-NEXT:.b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 51                                  // DW_AT_address_class
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 2                                   // DW_AT_location
; CHECK-NEXT:.b8 10                                  // DW_FORM_block1
; CHECK-NEXT:.b8 49                                  // DW_AT_abstract_origin
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 30                                  // Abbreviation Code
; CHECK-NEXT:.b8 57                                  // DW_TAG_namespace
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 31                                  // Abbreviation Code
; CHECK-NEXT:.b8 8                                   // DW_TAG_imported_declaration
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 24                                  // DW_AT_import
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 32                                  // Abbreviation Code
; CHECK-NEXT:.b8 8                                   // DW_TAG_imported_declaration
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 24                                  // DW_AT_import
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 33                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 34                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 35                                  // Abbreviation Code
; CHECK-NEXT:.b8 22                                  // DW_TAG_typedef
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 36                                  // Abbreviation Code
; CHECK-NEXT:.b8 19                                  // DW_TAG_structure_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 37                                  // Abbreviation Code
; CHECK-NEXT:.b8 19                                  // DW_TAG_structure_type
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 11                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 38                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 135                                 // DW_AT_noreturn
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 39                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 40                                  // Abbreviation Code
; CHECK-NEXT:.b8 21                                  // DW_TAG_subroutine_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 41                                  // Abbreviation Code
; CHECK-NEXT:.b8 15                                  // DW_TAG_pointer_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 42                                  // Abbreviation Code
; CHECK-NEXT:.b8 38                                  // DW_TAG_const_type
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 43                                  // Abbreviation Code
; CHECK-NEXT:.b8 21                                  // DW_TAG_subroutine_type
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 44                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 135                                 // DW_AT_noreturn
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 45                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 63                                  // DW_AT_external
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 46                                  // Abbreviation Code
; CHECK-NEXT:.b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT:.b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT:.b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 64
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 3                                   // DW_AT_name
; CHECK-NEXT:.b8 8                                   // DW_FORM_string
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_FORM_data1
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5                                   // DW_FORM_data2
; CHECK-NEXT:.b8 73                                  // DW_AT_type
; CHECK-NEXT:.b8 19                                  // DW_FORM_ref4
; CHECK-NEXT:.b8 60                                  // DW_AT_declaration
; CHECK-NEXT:.b8 12                                  // DW_FORM_flag
; CHECK-NEXT:.b8 0                                   // EOM(1)
; CHECK-NEXT:.b8 0                                   // EOM(2)
; CHECK-NEXT:.b8 0                                   // EOM(3)
; CHECK-NEXT:	}
; CHECK-NEXT:	.section	.debug_info
; CHECK-NEXT:	{
; CHECK-NEXT:.b32 10035                              // Length of Unit
; CHECK-NEXT:.b8 2                                   // DWARF version number
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 .debug_abbrev                      // Offset Into Abbrev. Section
; CHECK-NEXT:.b8 8                                   // Address Size (in bytes)
; CHECK-NEXT:.b8 1                                   // Abbrev [1] 0xb:0x272c DW_TAG_compile_unit
; CHECK-NEXT:.b8 0                                   // DW_AT_producer
; CHECK-NEXT:.b8 4                                   // DW_AT_language
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 45
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 46
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 .debug_line                        // DW_AT_stmt_list
; CHECK-NEXT:.b8 47                                  // DW_AT_comp_dir
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 47
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // Abbrev [2] 0x31:0x22a DW_TAG_structure_type
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 77                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x4f:0x4f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 78                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x9e:0x4f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 79                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0xed:0x4f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 80                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 4                                   // Abbrev [4] 0x13c:0x49 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 83                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 619                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x17e:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 666                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 6                                   // Abbrev [6] 0x185:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x1a5:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 676                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 6                                   // Abbrev [6] 0x1ac:0x2c DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x1cc:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 676                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 681                                // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 8                                   // Abbrev [8] 0x1d8:0x43 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 82
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 61
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x20f:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 666                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x215:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 681                                // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 9                                   // Abbrev [9] 0x21b:0x3f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 38
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 686                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x253:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 666                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x25b:0x10 DW_TAG_base_type
; CHECK-NEXT:.b8 117                                 // DW_AT_name
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_encoding
; CHECK-NEXT:.b8 4                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 2                                   // Abbrev [2] 0x26b:0x2f DW_TAG_structure_type
; CHECK-NEXT:.b8 117                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 190                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x275:0xc DW_TAG_member
; CHECK-NEXT:.b8 120                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 192                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x281:0xc DW_TAG_member
; CHECK-NEXT:.b8 121                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 192                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x28d:0xc DW_TAG_member
; CHECK-NEXT:.b8 122                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 192                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 8
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x29a:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 671                                // DW_AT_type
; CHECK-NEXT:.b8 13                                  // Abbrev [13] 0x29f:0x5 DW_TAG_const_type
; CHECK-NEXT:.b32 49                                 // DW_AT_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x2a4:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 49                                 // DW_AT_type
; CHECK-NEXT:.b8 14                                  // Abbrev [14] 0x2a9:0x5 DW_TAG_reference_type
; CHECK-NEXT:.b32 671                                // DW_AT_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x2ae:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 49                                 // DW_AT_type
; CHECK-NEXT:.b8 15                                  // Abbrev [15] 0x2b3:0x6 DW_TAG_subprogram
; CHECK-NEXT:.b32 79                                 // DW_AT_specification
; CHECK-NEXT:.b8 1                                   // DW_AT_inline
; CHECK-NEXT:.b8 2                                   // Abbrev [2] 0x2b9:0x228 DW_TAG_structure_type
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 88                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x2d7:0x4f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 89                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x326:0x4f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 90                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x375:0x4f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 91                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 4                                   // Abbrev [4] 0x3c4:0x47 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 94                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 1249                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x404:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1425                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 6                                   // Abbrev [6] 0x40b:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 96                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x42b:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1435                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 6                                   // Abbrev [6] 0x432:0x2c DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 96                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x452:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1435                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x458:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1440                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 8                                   // Abbrev [8] 0x45e:0x43 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 82
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 61
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 96                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x495:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1425                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x49b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1440                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 9                                   // Abbrev [9] 0x4a1:0x3f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 107
; CHECK-NEXT:.b8 68
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 38
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 96                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 1445                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x4d9:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1425                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 16                                  // Abbrev [16] 0x4e1:0x9d DW_TAG_structure_type
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 161                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 17                                  // Abbrev [17] 0x4eb:0xd DW_TAG_member
; CHECK-NEXT:.b8 120                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 163                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 17                                  // Abbrev [17] 0x4f8:0xd DW_TAG_member
; CHECK-NEXT:.b8 121                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 163                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b8 17                                  // Abbrev [17] 0x505:0xd DW_TAG_member
; CHECK-NEXT:.b8 122                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 163                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 8
; CHECK-NEXT:.b8 18                                  // Abbrev [18] 0x512:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 165                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x51d:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1406                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x523:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x528:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x52d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 18                                  // Abbrev [18] 0x533:0x17 DW_TAG_subprogram
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 166                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x53e:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1406                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x544:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1411                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 19                                  // Abbrev [19] 0x54a:0x33 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 167                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 1411                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x576:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 1406                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x57e:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 1249                               // DW_AT_type
; CHECK-NEXT:.b8 20                                  // Abbrev [20] 0x583:0xe DW_TAG_typedef
; CHECK-NEXT:.b32 619                                // DW_AT_type
; CHECK-NEXT:.b8 117                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 127                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x591:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 1430                               // DW_AT_type
; CHECK-NEXT:.b8 13                                  // Abbrev [13] 0x596:0x5 DW_TAG_const_type
; CHECK-NEXT:.b32 697                                // DW_AT_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x59b:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 697                                // DW_AT_type
; CHECK-NEXT:.b8 14                                  // Abbrev [14] 0x5a0:0x5 DW_TAG_reference_type
; CHECK-NEXT:.b32 1430                               // DW_AT_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x5a5:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 697                                // DW_AT_type
; CHECK-NEXT:.b8 15                                  // Abbrev [15] 0x5aa:0x6 DW_TAG_subprogram
; CHECK-NEXT:.b32 727                                // DW_AT_specification
; CHECK-NEXT:.b8 1                                   // DW_AT_inline
; CHECK-NEXT:.b8 2                                   // Abbrev [2] 0x5b0:0x233 DW_TAG_structure_type
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 66                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x5cf:0x50 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 67                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x61f:0x50 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 68                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // Abbrev [3] 0x66f:0x50 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 69                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 4                                   // Abbrev [4] 0x6bf:0x4a DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 72                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 619                                // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x702:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2019                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 6                                   // Abbrev [6] 0x709:0x28 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 74                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x72a:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2029                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 6                                   // Abbrev [6] 0x731:0x2d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 74                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x752:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2029                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x758:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2034                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 8                                   // Abbrev [8] 0x75e:0x44 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 82
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 61
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 74                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x796:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2019                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x79c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2034                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 9                                   // Abbrev [9] 0x7a2:0x40 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 73
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 111                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 38
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 74                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2039                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                        // DW_ACCESS_private
; CHECK-NEXT:.b8 5                                   // Abbrev [5] 0x7db:0x6 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2019                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_artificial
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x7e3:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 2024                               // DW_AT_type
; CHECK-NEXT:.b8 13                                  // Abbrev [13] 0x7e8:0x5 DW_TAG_const_type
; CHECK-NEXT:.b32 1456                               // DW_AT_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x7ed:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 1456                               // DW_AT_type
; CHECK-NEXT:.b8 14                                  // Abbrev [14] 0x7f2:0x5 DW_TAG_reference_type
; CHECK-NEXT:.b32 2024                               // DW_AT_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x7f7:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 1456                               // DW_AT_type
; CHECK-NEXT:.b8 15                                  // Abbrev [15] 0x7fc:0x6 DW_TAG_subprogram
; CHECK-NEXT:.b32 1487                               // DW_AT_specification
; CHECK-NEXT:.b8 1                                   // DW_AT_inline
; CHECK-NEXT:.b8 21                                  // Abbrev [21] 0x802:0x32 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 1                                   // DW_AT_inline
; CHECK-NEXT:.b8 22                                  // Abbrev [22] 0x816:0x9 DW_TAG_formal_parameter
; CHECK-NEXT:.b8 120                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 22                                  // Abbrev [22] 0x81f:0x9 DW_TAG_formal_parameter
; CHECK-NEXT:.b8 121                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 22                                  // Abbrev [22] 0x828:0xb DW_TAG_formal_parameter
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 2109                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x834:0x9 DW_TAG_base_type
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_encoding
; CHECK-NEXT:.b8 4                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x83d:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 23                                  // Abbrev [23] 0x842:0xd5 DW_TAG_subprogram
; CHECK-NEXT:.b64 $L__func_begin0                    // DW_AT_low_pc
; CHECK-NEXT:.b64 $L__func_end0                      // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_AT_frame_base
; CHECK-NEXT:.b8 156
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 83
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 24                                  // Abbrev [24] 0x86d:0x10 DW_TAG_formal_parameter
; CHECK-NEXT:.b8 2                                   // DW_AT_address_class
; CHECK-NEXT:.b8 5                                   // DW_AT_location
; CHECK-NEXT:.b8 144
; CHECK-NEXT:.b8 178
; CHECK-NEXT:.b8 228
; CHECK-NEXT:.b8 149
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 25                                  // Abbrev [25] 0x87d:0xd DW_TAG_formal_parameter
; CHECK-NEXT:.b32 $L__debug_loc0                     // DW_AT_location
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 22                                  // Abbrev [22] 0x88a:0x9 DW_TAG_formal_parameter
; CHECK-NEXT:.b8 120                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 2109                               // DW_AT_type
; CHECK-NEXT:.b8 22                                  // Abbrev [22] 0x893:0x9 DW_TAG_formal_parameter
; CHECK-NEXT:.b8 121                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 2109                               // DW_AT_type
; CHECK-NEXT:.b8 26                                  // Abbrev [26] 0x89c:0xd DW_TAG_variable
; CHECK-NEXT:.b32 $L__debug_loc1                     // DW_AT_location
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_line
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 27                                  // Abbrev [27] 0x8a9:0x18 DW_TAG_inlined_subroutine
; CHECK-NEXT:.b32 691                                // DW_AT_abstract_origin
; CHECK-NEXT:.b64 $L__tmp1                           // DW_AT_low_pc
; CHECK-NEXT:.b64 $L__tmp2                           // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_AT_call_file
; CHECK-NEXT:.b8 6                                   // DW_AT_call_line
; CHECK-NEXT:.b8 11                                  // DW_AT_call_column
; CHECK-NEXT:.b8 27                                  // Abbrev [27] 0x8c1:0x18 DW_TAG_inlined_subroutine
; CHECK-NEXT:.b32 1450                               // DW_AT_abstract_origin
; CHECK-NEXT:.b64 $L__tmp2                           // DW_AT_low_pc
; CHECK-NEXT:.b64 $L__tmp3                           // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_AT_call_file
; CHECK-NEXT:.b8 6                                   // DW_AT_call_line
; CHECK-NEXT:.b8 24                                  // DW_AT_call_column
; CHECK-NEXT:.b8 27                                  // Abbrev [27] 0x8d9:0x18 DW_TAG_inlined_subroutine
; CHECK-NEXT:.b32 2044                               // DW_AT_abstract_origin
; CHECK-NEXT:.b64 $L__tmp3                           // DW_AT_low_pc
; CHECK-NEXT:.b64 $L__tmp4                           // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_AT_call_file
; CHECK-NEXT:.b8 6                                   // DW_AT_call_line
; CHECK-NEXT:.b8 37                                  // DW_AT_call_column
; CHECK-NEXT:.b8 28                                  // Abbrev [28] 0x8f1:0x25 DW_TAG_inlined_subroutine
; CHECK-NEXT:.b32 2050                               // DW_AT_abstract_origin
; CHECK-NEXT:.b64 $L__tmp9                           // DW_AT_low_pc
; CHECK-NEXT:.b64 $L__tmp10                          // DW_AT_high_pc
; CHECK-NEXT:.b8 1                                   // DW_AT_call_file
; CHECK-NEXT:.b8 8                                   // DW_AT_call_line
; CHECK-NEXT:.b8 5                                   // DW_AT_call_column
; CHECK-NEXT:.b8 29                                  // Abbrev [29] 0x909:0xc DW_TAG_formal_parameter
; CHECK-NEXT:.b8 2                                   // DW_AT_address_class
; CHECK-NEXT:.b8 5                                   // DW_AT_location
; CHECK-NEXT:.b8 144
; CHECK-NEXT:.b8 179
; CHECK-NEXT:.b8 204
; CHECK-NEXT:.b8 149
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 2079                               // DW_AT_abstract_origin
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 30                                  // Abbrev [30] 0x917:0x588 DW_TAG_namespace
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x91c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 202                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3743                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x923:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 203                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3787                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x92a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 204                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3816                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x931:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 205                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3847                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x938:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 206                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3876                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x93f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 207                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3907                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x946:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 208                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3936                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x94d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 209                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3973                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x954:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 210                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4004                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x95b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 211                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4033                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x962:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 212                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4062                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x969:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 213                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4105                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x970:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 214                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4132                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x977:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 215                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4161                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x97e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 216                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4188                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x985:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 217                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4217                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x98c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 218                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4244                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x993:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 219                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4273                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x99a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 220                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4304                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9a1:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 221                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4333                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9a8:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 222                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4368                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9af:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 223                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4399                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9b6:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 224                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4438                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9bd:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 225                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4473                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9c4:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 226                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4508                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9cb:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 227                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4543                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9d2:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 228                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4592                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9d9:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 229                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4635                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9e0:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 230                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4672                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9e7:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 231                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4703                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9ee:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 232                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4748                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9f5:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 233                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4793                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x9fc:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 234                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4849                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa03:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 235                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4880                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa0a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 236                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4919                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa11:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 237                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4969                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa18:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 238                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5023                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa1f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 239                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5054                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa26:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 240                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5091                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa2d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 241                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5141                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa34:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 242                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5182                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa3b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 243                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5219                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa42:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 244                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5252                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa49:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 245                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5283                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa50:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 246                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5316                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa57:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 247                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5343                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa5e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 248                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5374                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa65:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 249                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5405                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa6c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 250                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5434                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa73:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 251                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5463                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa7a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 252                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5494                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa81:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 253                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5527                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa88:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 254                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5562                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xa8f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 255                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5598                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xa96:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 0                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5655                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xa9e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 1                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5686                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xaa6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 2                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5725                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xaae:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5770                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xab6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5803                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xabe:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5848                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xac6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5894                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xace:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5923                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xad6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5954                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xade:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 9                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5995                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xae6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 10                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6034                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xaee:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 11                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6069                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xaf6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6096                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xafe:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6125                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb06:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6154                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb0e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 15                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6181                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb16:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 16                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6210                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb1e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 17                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6243                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb26:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 102                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6274                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb2d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 121                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6294                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb34:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 140                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6314                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb3b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 159                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6334                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb42:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 180                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6360                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb49:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 199                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6380                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb50:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 218                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6399                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xb57:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 237                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6419                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb5e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 0                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6438                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb66:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 19                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6458                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb6e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 38                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6479                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb76:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 59                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6504                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb7e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 78                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6530                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb86:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 97                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6556                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb8e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 116                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6575                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb96:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 135                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6596                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xb9e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 147                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6626                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xba6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 184                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6650                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xbae:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 203                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6669                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xbb6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 222                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6689                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xbbe:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 241                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6709                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xbc6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 6                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 6728                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbce:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 118                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6748                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbd5:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 119                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6763                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbdc:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 121                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6811                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbe3:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 122                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6824                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbea:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 123                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6844                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbf1:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 129                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6873                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbf8:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 130                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6893                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xbff:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 131                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6914                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc06:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 132                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6935                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc0d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 133                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7063                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc14:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 134                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7091                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc1b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 135                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7116                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc22:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 136                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7134                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc29:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 137                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7151                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc30:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 138                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7179                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc37:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 139                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7200                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc3e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 140                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7226                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc45:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 142                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7249                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc4c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 143                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7276                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc53:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 144                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7327                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc5a:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 146                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7360                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc61:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 152                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7393                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc68:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 153                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7408                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc6f:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 154                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7437                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc76:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 155                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7455                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc7d:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 156                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7487                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc84:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 157                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7519                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc8b:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 158                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7552                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc92:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 160                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7575                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xc99:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 161                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7620                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xca0:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 241                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7768                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xca7:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 243                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7817                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcae:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 245                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7836                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcb5:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 246                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7722                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcbc:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 247                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7858                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcc3:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 249                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7885                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcca:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 250                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 8000                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcd1:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 251                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7907                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcd8:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 252                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7940                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0xcdf:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 253                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 8027                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xce6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 149                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8070                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xcee:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 150                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8102                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xcf6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 151                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8136                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xcfe:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 152                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8168                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd06:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 153                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8202                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd0e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 154                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8242                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd16:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 155                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8274                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd1e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 156                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8308                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd26:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 157                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8340                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd2e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 158                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8372                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd36:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 159                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8418                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd3e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 160                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8448                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd46:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 161                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8480                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd4e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 162                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8512                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd56:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 163                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8542                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd5e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 164                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8574                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd66:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 165                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8604                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd6e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 166                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8638                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd76:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 167                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8670                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd7e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 168                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8708                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd86:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 169                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8742                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd8e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 170                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8784                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd96:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 171                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8822                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xd9e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 172                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8860                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xda6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 173                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8898                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdae:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 174                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8939                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdb6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 175                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 8979                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdbe:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 176                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9013                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdc6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 177                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9053                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdce:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 178                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9089                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdd6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 179                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9125                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdde:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 180                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9163                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xde6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 181                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9197                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdee:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 182                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9231                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdf6:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 183                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9263                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xdfe:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 184                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9295                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe06:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 185                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9325                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe0e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 186                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9359                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe16:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 187                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9395                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe1e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 188                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9434                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe26:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 189                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9477                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe2e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 190                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9526                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe36:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 191                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9562                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe3e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 192                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9611                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe46:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 193                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9660                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe4e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 194                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9692                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe56:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 195                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9726                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe5e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 196                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9770                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe66:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 197                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9812                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe6e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 198                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9842                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe76:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 199                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9874                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe7e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 200                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9906                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe86:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 201                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9936                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe8e:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 202                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 9968                               // DW_AT_import
; CHECK-NEXT:.b8 32                                  // Abbrev [32] 0xe96:0x8 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 13                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 203                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 10004                              // DW_AT_import
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xe9f:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 44                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xeb4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0xeba:0x11 DW_TAG_base_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_encoding
; CHECK-NEXT:.b8 8                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xecb:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 46                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xee2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xee8:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 48                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf01:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xf07:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 50                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf1e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xf24:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 52                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf3d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xf43:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 56                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf5a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xf60:0x25 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 54                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf7a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf7f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xf85:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xf9e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xfa4:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 60                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xfbb:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xfc1:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 62                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xfd8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0xfde:0x2b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 56
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 64                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0xffe:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1003:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1009:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 66                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x101e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1024:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 68                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x103b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1041:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 72                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1056:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x105c:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 70                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1073:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1079:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 76                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x108e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1094:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 74                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x10ab:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x10b1:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 78                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x10ca:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x10d0:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 80                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x10e7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x10ed:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 82                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1105:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x110a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1110:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 84                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1129:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x112f:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 86                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1146:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x114b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1150:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1156:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 88                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x116e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1173:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1179:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 90                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1191:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1196:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x119c:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 92                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x11b4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x11b9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x11bf:0x2a DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 94                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x11e3:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x11e9:0x7 DW_TAG_base_type
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_encoding
; CHECK-NEXT:.b8 4                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x11f0:0x26 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 96                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x120b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1210:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4630                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1216:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x121b:0x25 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 104                                 // DW_AT_name
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 98                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1235:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x123a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1240:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 100                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1259:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x125f:0x25 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 56
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 102                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x127e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x1284:0x8 DW_TAG_base_type
; CHECK-NEXT:.b8 98                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 2                                   // DW_AT_encoding
; CHECK-NEXT:.b8 1                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x128c:0x2d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 57
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 106                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x12ae:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x12b3:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x12b9:0x38 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 105                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x12e6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x12eb:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x12f1:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 108                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x130a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1310:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 112                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x132c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1331:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1337:0x32 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 111                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x135e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1363:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1369:0x36 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 114                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1394:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1399:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x139f:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 116                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x13b8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x13be:0x25 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 56
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 118                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x13dd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x13e3:0x32 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 120                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x140a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x140f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1415:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 121                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x142c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x1432:0xc DW_TAG_base_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_encoding
; CHECK-NEXT:.b8 8                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x143e:0x25 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 123                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1458:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x145d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1463:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 125                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x147e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1484:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 126                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x149d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x14a3:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 128                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x14be:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x14c4:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 138                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x14d9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x14df:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 130                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x14f8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x14fe:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 132                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1517:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x151d:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 134                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1534:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x153a:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 136                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1551:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1557:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 140                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1570:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1576:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 142                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1591:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1597:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 143                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x15b4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x15ba:0x24 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 145                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x15d3:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x15d8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2109                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x15de:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 146                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x15f5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x15fb:0xa DW_TAG_base_type
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_encoding
; CHECK-NEXT:.b8 8                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1605:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 5642                               // DW_AT_type
; CHECK-NEXT:.b8 13                                  // Abbrev [13] 0x160a:0x5 DW_TAG_const_type
; CHECK-NEXT:.b32 5647                               // DW_AT_type
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x160f:0x8 DW_TAG_base_type
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 8                                   // DW_AT_encoding
; CHECK-NEXT:.b8 1                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1617:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 75
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 147                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1630:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1636:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 57
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 149                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1657:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x165d:0x2d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 57
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 151                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x167f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1684:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x168a:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 112                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 155                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x16a0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x16a5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x16ab:0x2d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 57
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 157                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x16cd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x16d2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x16d8:0x2e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 159                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x16f6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x16fb:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1700:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4630                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1706:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 161                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x171d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1723:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 163                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x173c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1742:0x29 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 165                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1760:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1765:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x176b:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 167                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1787:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x178c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1792:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 169                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 4740                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x17af:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x17b5:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 171                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x17ca:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x17d0:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 173                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x17e7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x17ed:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 175                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1804:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x180a:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 177                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x181f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1825:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 179                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x183c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1842:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 181                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x185d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 33                                  // Abbrev [33] 0x1863:0x1f DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 183                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x187c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1882:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 54                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1890:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1896:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 56                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x18a4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x18aa:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 58                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x18b8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x18be:0x1a DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 60                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x18cd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x18d2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x18d8:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 178                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x18e6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x18ec:0x13 DW_TAG_subprogram
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 63                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x18f9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x18ff:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 72                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x190d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1913:0x13 DW_TAG_subprogram
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 100                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1920:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1926:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 181                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1934:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x193a:0x15 DW_TAG_subprogram
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 184                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1949:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x194f:0x19 DW_TAG_subprogram
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 187                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x195d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1962:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1968:0x1a DW_TAG_subprogram
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 103                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1977:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x197c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4630                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1982:0x1a DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 106                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1991:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1996:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x199c:0x13 DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 109                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x19a9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x19af:0x15 DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 112                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x19be:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x19c4:0x19 DW_TAG_subprogram
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 115                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x19d2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x19d7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6621                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x19dd:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x19e2:0x18 DW_TAG_subprogram
; CHECK-NEXT:.b8 112                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 153                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x19ef:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x19f4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x19fa:0x13 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 65                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1a07:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1a0d:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 74                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1a1b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1a21:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 156                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1a2f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1a35:0x13 DW_TAG_subprogram
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 67                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1a42:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1a48:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 76                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1a56:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 35                                  // Abbrev [35] 0x1a5c:0xd DW_TAG_typedef
; CHECK-NEXT:.b32 6761                               // DW_AT_type
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 101                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 36                                  // Abbrev [36] 0x1a69:0x2 DW_TAG_structure_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 35                                  // Abbrev [35] 0x1a6b:0xe DW_TAG_typedef
; CHECK-NEXT:.b32 6777                               // DW_AT_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 109                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 37                                  // Abbrev [37] 0x1a79:0x22 DW_TAG_structure_type
; CHECK-NEXT:.b8 16                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 105                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x1a7d:0xf DW_TAG_member
; CHECK-NEXT:.b8 113                                 // DW_AT_name
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 107                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x1a8c:0xe DW_TAG_member
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 108                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 8
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 38                                  // Abbrev [38] 0x1a9b:0xd DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 3                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 1                                   // DW_AT_noreturn
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1aa8:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ab6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1abc:0x17 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1acd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6867                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1ad3:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 6872                               // DW_AT_type
; CHECK-NEXT:.b8 40                                  // Abbrev [40] 0x1ad8:0x1 DW_TAG_subroutine_type
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1ad9:0x14 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 9                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 26                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ae7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1aed:0x15 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 22                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1afc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1b02:0x15 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 27                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b11:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1b17:0x2b DW_TAG_subprogram
; CHECK-NEXT:.b8 98                                  // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 10                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 20                                  // DW_AT_decl_line
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b28:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6979                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b2d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6979                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b32:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b37:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b3c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7020                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 41                                  // Abbrev [41] 0x1b42:0x1 DW_TAG_pointer_type
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1b43:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 6984                               // DW_AT_type
; CHECK-NEXT:.b8 42                                  // Abbrev [42] 0x1b48:0x1 DW_TAG_const_type
; CHECK-NEXT:.b8 35                                  // Abbrev [35] 0x1b49:0xe DW_TAG_typedef
; CHECK-NEXT:.b32 6999                               // DW_AT_type
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 122
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 11                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 62                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x1b57:0x15 DW_TAG_base_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_encoding
; CHECK-NEXT:.b8 8                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 20                                  // Abbrev [20] 0x1b6c:0x16 DW_TAG_typedef
; CHECK-NEXT:.b32 7042                               // DW_AT_type
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 230                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1b82:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 7047                               // DW_AT_type
; CHECK-NEXT:.b8 43                                  // Abbrev [43] 0x1b87:0x10 DW_TAG_subroutine_type
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b8c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6979                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1b91:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6979                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1b97:0x1c DW_TAG_subprogram
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 212                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ba8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1bad:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1bb3:0x19 DW_TAG_subprogram
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 21                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 6748                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1bc1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1bc6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 44                                  // Abbrev [44] 0x1bcc:0x12 DW_TAG_subprogram
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 31                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 1                                   // DW_AT_noreturn
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1bd8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 18                                  // Abbrev [18] 0x1bde:0x11 DW_TAG_subprogram
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 227                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1be9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1bef:0x17 DW_TAG_subprogram
; CHECK-NEXT:.b8 103                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 52                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 7174                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c00:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1c06:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 5647                               // DW_AT_type
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1c0b:0x15 DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c1a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1c20:0x1a DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 23                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 6763                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c2f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c34:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1c3a:0x17 DW_TAG_subprogram
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 210                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c4b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1c51:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 95                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c61:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c66:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1c6c:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 106                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c7f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7311                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c84:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1c89:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1c8f:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 7316                               // DW_AT_type
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x1c94:0xb DW_TAG_base_type
; CHECK-NEXT:.b8 119                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 5                                   // DW_AT_encoding
; CHECK-NEXT:.b8 4                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1c9f:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 98                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1cb0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7311                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1cb5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1cba:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 18                                  // Abbrev [18] 0x1cc0:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 113                                 // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 253                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ccc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1cd1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1cd6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1cdb:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7020                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 45                                  // Abbrev [45] 0x1ce1:0xf DW_TAG_subprogram
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 118                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1cf0:0x1d DW_TAG_subprogram
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 224                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d02:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6978                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d07:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 18                                  // Abbrev [18] 0x1d0d:0x12 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 120                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d19:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 603                                // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1d1f:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 164                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5627                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d2f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d34:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1d3a:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 7174                               // DW_AT_type
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1d3f:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 183                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d4f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d54:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d59:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1d5f:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 187                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 6999                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d70:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d75:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d7a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1d80:0x17 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 205                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1d91:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1d97:0x23 DW_TAG_subprogram
; CHECK-NEXT:.b8 119                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 109                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1daa:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7174                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1daf:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7610                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1db4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 6985                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 12                                  // Abbrev [12] 0x1dba:0x5 DW_TAG_pointer_type
; CHECK-NEXT:.b32 7615                               // DW_AT_type
; CHECK-NEXT:.b8 13                                  // Abbrev [13] 0x1dbf:0x5 DW_TAG_const_type
; CHECK-NEXT:.b32 7316                               // DW_AT_type
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1dc4:0x1c DW_TAG_subprogram
; CHECK-NEXT:.b8 119                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 102                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1dd5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7174                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1dda:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7316                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 30                                  // Abbrev [30] 0x1de0:0x78 DW_TAG_namespace
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1deb:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 201                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7768                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1df2:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 207                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7817                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1df9:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 211                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7836                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1e00:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 217                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7858                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1e07:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 228                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7885                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1e0e:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 229                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7907                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1e15:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 230                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7940                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1e1c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 232                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 8000                               // DW_AT_import
; CHECK-NEXT:.b8 31                                  // Abbrev [31] 0x1e23:0x7 DW_TAG_imported_declaration
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 233                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 8027                               // DW_AT_import
; CHECK-NEXT:.b8 4                                   // Abbrev [4] 0x1e2a:0x2d DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 78
; CHECK-NEXT:.b8 57
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 51
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 100                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 8                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 214                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7768                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1e4c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1e51:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 35                                  // Abbrev [35] 0x1e58:0xf DW_TAG_typedef
; CHECK-NEXT:.b32 7783                               // DW_AT_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 95
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 121                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 37                                  // Abbrev [37] 0x1e67:0x22 DW_TAG_structure_type
; CHECK-NEXT:.b8 16                                  // DW_AT_byte_size
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 117                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x1e6b:0xf DW_TAG_member
; CHECK-NEXT:.b8 113                                 // DW_AT_name
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 119                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 11                                  // Abbrev [11] 0x1e7a:0xe DW_TAG_member
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 120                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT:.b8 35
; CHECK-NEXT:.b8 8
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 44                                  // Abbrev [44] 0x1e89:0x13 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_name
; CHECK-NEXT:.b8 69
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 45                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 1                                   // DW_AT_noreturn
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1e96:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1e9c:0x16 DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1eac:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1eb2:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 118
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 29                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 7768                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ec2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ec7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 39                                  // Abbrev [39] 0x1ecd:0x16 DW_TAG_subprogram
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 36                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 1
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1edd:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1ee3:0x21 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 209                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ef4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1ef9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1efe:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1f04:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 214                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 7974                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f16:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f1b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f20:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x1f26:0x1a DW_TAG_base_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_encoding
; CHECK-NEXT:.b8 8                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1f40:0x1b DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 172                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f50:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f55:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 34                                  // Abbrev [34] 0x1f5b:0x1c DW_TAG_subprogram
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_file
; CHECK-NEXT:.b8 175                                 // DW_AT_decl_line
; CHECK-NEXT:.b32 8055                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 1                                   // DW_AT_external
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f6c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5637                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1f71:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 7482                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 10                                  // Abbrev [10] 0x1f77:0xf DW_TAG_base_type
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 32
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 4                                   // DW_AT_encoding
; CHECK-NEXT:.b8 8                                   // DW_AT_byte_size
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x1f86:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 62                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1fa0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x1fa6:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 90                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1fc2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x1fc8:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 57                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x1fe2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x1fe8:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 95                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2004:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x200a:0x28 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 47                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2027:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x202c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2032:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 52                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x204c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2052:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 97                                  // DW_AT_name
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 100                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x206e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2074:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 150                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x208e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2094:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 155                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x20ae:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x20b4:0x2e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 57
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 165                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x20d7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x20dc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x20e2:0x1e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 219                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x20fa:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2100:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 99                                  // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 32                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x211a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2120:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 210                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x213a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2140:0x1e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 200                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2158:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x215e:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 145                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2178:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x217e:0x1e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2196:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x219c:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 101                                 // DW_AT_name
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 105                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x21b8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x21be:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 95                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x21d8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x21de:0x26 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 80                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x21f9:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x21fe:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2204:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2220:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2226:0x2a DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 32                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2240:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2245:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x224a:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2250:0x26 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 110                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x226b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2270:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2276:0x26 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 105                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2291:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2296:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x229c:0x26 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 17                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x22b7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x22bc:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x22c2:0x29 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 102                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 7                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x22e0:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x22e5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4630                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x22eb:0x28 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 104                                 // DW_AT_name
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 110                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2308:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x230d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2313:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 105                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x232f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2335:0x28 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 240                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2352:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2357:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x235d:0x24 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 235                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x237b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2381:0x24 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 125                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x239f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x23a5:0x26 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 56
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 66                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 3770                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x23c5:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x23cb:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 76                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x23e7:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x23ed:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 85                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2409:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x240f:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 50
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 5                                   // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2429:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x242f:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 90                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2449:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x244f:0x1e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 67                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2467:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x246d:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 116                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2489:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x248f:0x24 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 108                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 71                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x24ad:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x24b3:0x27 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 109                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x24cf:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x24d4:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2109                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x24da:0x2b DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 121
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 130                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x24ff:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2505:0x31 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 110                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 120
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 194                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x252b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2530:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2536:0x24 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 112
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 112                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 119
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 47                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x254f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2554:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x255a:0x31 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 49
; CHECK-NEXT:.b8 48
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 22                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2580:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2585:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x258b:0x31 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 80
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 101
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 27                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x25ac:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x25b1:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x25b6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4630                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x25bc:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 111                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x25d6:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x25dc:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 114                                 // DW_AT_name
; CHECK-NEXT:.b8 111
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 100
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 61                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x25f8:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x25fe:0x2c DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 56
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 250                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x261f:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2624:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 5170                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x262a:0x2a DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 108
; CHECK-NEXT:.b8 98
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 245                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2649:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x264e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 4585                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2654:0x1e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 210                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x266c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2672:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 105
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 37                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x268c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2692:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 115
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 115                                 // DW_AT_name
; CHECK-NEXT:.b8 113
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 139                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 3
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x26ac:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x26b2:0x1e DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 52
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 252                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 4
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x26ca:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x26d0:0x20 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 53
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 104
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 42                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 5
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x26ea:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x26f0:0x24 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 55
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 103
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 109
; CHECK-NEXT:.b8 97
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 12                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 56                                  // DW_AT_decl_line
; CHECK-NEXT:.b8 6
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x270e:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 46                                  // Abbrev [46] 0x2714:0x22 DW_TAG_subprogram
; CHECK-NEXT:.b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT:.b8 90
; CHECK-NEXT:.b8 76
; CHECK-NEXT:.b8 54
; CHECK-NEXT:.b8 116
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 116                                 // DW_AT_name
; CHECK-NEXT:.b8 114
; CHECK-NEXT:.b8 117
; CHECK-NEXT:.b8 110
; CHECK-NEXT:.b8 99
; CHECK-NEXT:.b8 102
; CHECK-NEXT:.b8 0
; CHECK-NEXT:.b8 14                                  // DW_AT_decl_file
; CHECK-NEXT:.b8 150                                 // DW_AT_decl_line
; CHECK-NEXT:.b8 2
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 1                                   // DW_AT_declaration
; CHECK-NEXT:.b8 7                                   // Abbrev [7] 0x2730:0x5 DW_TAG_formal_parameter
; CHECK-NEXT:.b32 2100                               // DW_AT_type
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:.b8 0                                   // End Of Children Mark
; CHECK-NEXT:	}
; CHECK-NEXT:	.section	.debug_macinfo	{	}
; CHECK-NOT: debug_

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
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
