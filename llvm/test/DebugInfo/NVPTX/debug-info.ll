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

; CHECK-DAG: .file [[DEBUG_INFO_CU]] "{{.*}}debug-info.cu"
; CHECK-DAG: .file [[BUILTUIN_VARS_H]] "{{.*}}clang/include{{/|\\\\}}__clang_cuda_builtin_vars.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}clang/include{{/|\\\\}}__clang_cuda_math_forward_declares.h"
; CHECK-DAG: .file {{[0-9]+}} "{{.*}}/usr/local/cuda/include{{/|\\\\}}vector_types.h"

; CHECK:	.section	.debug_loc
; CHECK-NEXT: 	{
; CHECK-NEXT: $L__debug_loc0:
; CHECK-NEXT: .b64 $L__tmp8
; CHECK-NEXT: .b64 $L__tmp10
; CHECK-NEXT: .b8 5                                   // Loc expr size
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 144                                 // DW_OP_regx
; CHECK-NEXT: .b8 177                                 // 2450993
; CHECK-NEXT: .b8 204                                 // 
; CHECK-NEXT: .b8 149                                 // 
; CHECK-NEXT: .b8 1                                   // 
; CHECK-NEXT: .b64 0
; CHECK-NEXT: .b64 0
; CHECK-NEXT: $L__debug_loc1:
; CHECK-NEXT: .b64 $L__tmp5
; CHECK-NEXT: .b64 $L__func_end0
; CHECK-NEXT: .b8 5                                   // Loc expr size
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 144                                 // DW_OP_regx
; CHECK-NEXT: .b8 177                                 // 2454065
; CHECK-NEXT: .b8 228                                 // 
; CHECK-NEXT: .b8 149                                 // 
; CHECK-NEXT: .b8 1                                   // 
; CHECK-NEXT: .b64 0
; CHECK-NEXT: .b64 0
; CHECK-NEXT: 	}
; CHECK-NEXT: 	.section	.debug_abbrev
; CHECK-NEXT: 	{
; CHECK-NEXT: .b8 1                                   // Abbreviation Code
; CHECK-NEXT: .b8 17                                  // DW_TAG_compile_unit
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 37                                  // DW_AT_producer
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 19                                  // DW_AT_language
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 16                                  // DW_AT_stmt_list
; CHECK-NEXT: .b8 6                                   // DW_FORM_data4
; CHECK-NEXT: .b8 27                                  // DW_AT_comp_dir
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 2                                   // Abbreviation Code
; CHECK-NEXT: .b8 19                                  // DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 11                                  // DW_AT_byte_size
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 3                                   // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 4                                   // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 5                                   // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 52                                  // DW_AT_artificial
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 6                                   // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 50                                  // DW_AT_accessibility
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 7                                   // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 8                                   // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 50                                  // DW_AT_accessibility
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 9                                   // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 50                                  // DW_AT_accessibility
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 10                                  // Abbreviation Code
; CHECK-NEXT: .b8 36                                  // DW_TAG_base_type
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 62                                  // DW_AT_encoding
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 11                                  // DW_AT_byte_size
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 11                                  // Abbreviation Code
; CHECK-NEXT: .b8 13                                  // DW_TAG_member
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 56                                  // DW_AT_data_member_location
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 12                                  // Abbreviation Code
; CHECK-NEXT: .b8 15                                  // DW_TAG_pointer_type
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 13                                  // Abbreviation Code
; CHECK-NEXT: .b8 38                                  // DW_TAG_const_type
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 14                                  // Abbreviation Code
; CHECK-NEXT: .b8 16                                  // DW_TAG_reference_type
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 15                                  // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 71                                  // DW_AT_specification
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 32                                  // DW_AT_inline
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 16                                  // Abbreviation Code
; CHECK-NEXT: .b8 19                                  // DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 11                                  // DW_AT_byte_size
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 17                                  // Abbreviation Code
; CHECK-NEXT: .b8 13                                  // DW_TAG_member
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 56                                  // DW_AT_data_member_location
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 18                                  // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 19                                  // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 20                                  // Abbreviation Code
; CHECK-NEXT: .b8 22                                  // DW_TAG_typedef
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 21                                  // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 32                                  // DW_AT_inline
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 22                                  // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 23                                  // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 17                                  // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 18                                  // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 64                                  // DW_AT_frame_base
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 24                                  // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 51                                  // DW_AT_address_class
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 2                                   // DW_AT_location
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 25                                  // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 2                                   // DW_AT_location
; CHECK-NEXT: .b8 6                                   // DW_FORM_data4
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 26                                  // Abbreviation Code
; CHECK-NEXT: .b8 52                                  // DW_TAG_variable
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 2                                   // DW_AT_location
; CHECK-NEXT: .b8 6                                   // DW_FORM_data4
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 27                                  // Abbreviation Code
; CHECK-NEXT: .b8 29                                  // DW_TAG_inlined_subroutine
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 49                                  // DW_AT_abstract_origin
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 17                                  // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 18                                  // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 88                                  // DW_AT_call_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 89                                  // DW_AT_call_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 87                                  // DW_AT_call_column
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 28                                  // Abbreviation Code
; CHECK-NEXT: .b8 29                                  // DW_TAG_inlined_subroutine
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 49                                  // DW_AT_abstract_origin
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 17                                  // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 18                                  // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 88                                  // DW_AT_call_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 89                                  // DW_AT_call_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 87                                  // DW_AT_call_column
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 29                                  // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 51                                  // DW_AT_address_class
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 2                                   // DW_AT_location
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 49                                  // DW_AT_abstract_origin
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 30                                  // Abbreviation Code
; CHECK-NEXT: .b8 57                                  // DW_TAG_namespace
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 31                                  // Abbreviation Code
; CHECK-NEXT: .b8 8                                   // DW_TAG_imported_declaration
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 24                                  // DW_AT_import
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 32                                  // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 60                                  // DW_AT_declaration
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 0                                   // EOM(3)
; CHECK-NEXT: 	}
; CHECK-NEXT: 	.section	.debug_info
; CHECK-NEXT: 	{
; CHECK-NEXT: .b32 2388                               // Length of Unit
; CHECK-NEXT: .b8 2                                   // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                      // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                   // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                   // Abbrev [1] 0xb:0x94d DW_TAG_compile_unit
; CHECK-NEXT: .b8 0                                   // DW_AT_producer
; CHECK-NEXT: .b8 4                                   // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 100                                 // DW_AT_name
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 45
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                        // DW_AT_stmt_list
; CHECK-NEXT: .b8 47                                  // DW_AT_comp_dir
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 47
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x31:0x22a DW_TAG_structure_type
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 77                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x4f:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 78                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x9e:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 79                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0xed:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 122
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 122
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 80                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 4                                   // Abbrev [4] 0x13c:0x49 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 83                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 619                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x17e:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 666                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0x185:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 85                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x1a5:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 676                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0x1ac:0x2c DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 85                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x1cc:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 676                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x1d2:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 681                                // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 8                                   // Abbrev [8] 0x1d8:0x43 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 82
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 61
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 85                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x20f:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 666                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x215:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 681                                // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 9                                   // Abbrev [9] 0x21b:0x3f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 38
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 85                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 686                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x253:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 666                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 10                                  // Abbrev [10] 0x25b:0x10 DW_TAG_base_type
; CHECK-NEXT: .b8 117                                 // DW_AT_name
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 7                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x26b:0x2f DW_TAG_structure_type
; CHECK-NEXT: .b8 117                                 // DW_AT_name
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                                  // DW_AT_byte_size
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 190                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // Abbrev [11] 0x275:0xc DW_TAG_member
; CHECK-NEXT: .b8 120                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 192                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 11                                  // Abbrev [11] 0x281:0xc DW_TAG_member
; CHECK-NEXT: .b8 121                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 192                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 11                                  // Abbrev [11] 0x28d:0xc DW_TAG_member
; CHECK-NEXT: .b8 122                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 192                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 8
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x29a:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 671                                // DW_AT_type
; CHECK-NEXT: .b8 13                                  // Abbrev [13] 0x29f:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 49                                 // DW_AT_type
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x2a4:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 49                                 // DW_AT_type
; CHECK-NEXT: .b8 14                                  // Abbrev [14] 0x2a9:0x5 DW_TAG_reference_type
; CHECK-NEXT: .b32 671                                // DW_AT_type
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x2ae:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 49                                 // DW_AT_type
; CHECK-NEXT: .b8 15                                  // Abbrev [15] 0x2b3:0x6 DW_TAG_subprogram
; CHECK-NEXT: .b32 79                                 // DW_AT_specification
; CHECK-NEXT: .b8 1                                   // DW_AT_inline
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x2b9:0x228 DW_TAG_structure_type
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 88                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x2d7:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 89                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x326:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 90                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x375:0x4f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 122
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 122
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 91                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 4                                   // Abbrev [4] 0x3c4:0x47 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 52
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 94                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 1249                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x404:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1425                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0x40b:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 96                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x42b:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1435                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0x432:0x2c DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 96                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x452:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1435                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x458:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1440                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 8                                   // Abbrev [8] 0x45e:0x43 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 82
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 61
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 96                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x495:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1425                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x49b:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1440                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 9                                   // Abbrev [9] 0x4a1:0x3f DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 38
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 96                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 1445                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x4d9:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1425                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 16                                  // Abbrev [16] 0x4e1:0x9d DW_TAG_structure_type
; CHECK-NEXT: .b8 100                                 // DW_AT_name
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                                  // DW_AT_byte_size
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 161                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 17                                  // Abbrev [17] 0x4eb:0xd DW_TAG_member
; CHECK-NEXT: .b8 120                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 163                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 17                                  // Abbrev [17] 0x4f8:0xd DW_TAG_member
; CHECK-NEXT: .b8 121                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 163                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 17                                  // Abbrev [17] 0x505:0xd DW_TAG_member
; CHECK-NEXT: .b8 122                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 163                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 2                                   // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 8
; CHECK-NEXT: .b8 18                                  // Abbrev [18] 0x512:0x21 DW_TAG_subprogram
; CHECK-NEXT: .b8 100                                 // DW_AT_name
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 165                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x51d:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1406                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x523:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x528:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x52d:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 18                                  // Abbrev [18] 0x533:0x17 DW_TAG_subprogram
; CHECK-NEXT: .b8 100                                 // DW_AT_name
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 166                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x53e:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1406                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x544:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1411                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 19                                  // Abbrev [19] 0x54a:0x33 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 52
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 167                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 1411                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x576:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 1406                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x57e:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 1249                               // DW_AT_type
; CHECK-NEXT: .b8 20                                  // Abbrev [20] 0x583:0xe DW_TAG_typedef
; CHECK-NEXT: .b32 619                                // DW_AT_type
; CHECK-NEXT: .b8 117                                 // DW_AT_name
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 127                                 // DW_AT_decl_line
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x591:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 1430                               // DW_AT_type
; CHECK-NEXT: .b8 13                                  // Abbrev [13] 0x596:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 697                                // DW_AT_type
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x59b:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 697                                // DW_AT_type
; CHECK-NEXT: .b8 14                                  // Abbrev [14] 0x5a0:0x5 DW_TAG_reference_type
; CHECK-NEXT: .b32 1430                               // DW_AT_type
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x5a5:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 697                                // DW_AT_type
; CHECK-NEXT: .b8 15                                  // Abbrev [15] 0x5aa:0x6 DW_TAG_subprogram
; CHECK-NEXT: .b32 727                                // DW_AT_specification
; CHECK-NEXT: .b8 1                                   // DW_AT_inline
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x5b0:0x233 DW_TAG_structure_type
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 66                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x5cf:0x50 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 67                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x61f:0x50 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 68                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x66f:0x50 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 122
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 122
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 69                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 603                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 4                                   // Abbrev [4] 0x6bf:0x4a DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 72                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 619                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x702:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2019                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0x709:0x28 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 74                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x72a:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2029                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0x731:0x2d DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_name
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 74                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x752:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2029                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x758:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2034                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 8                                   // Abbrev [8] 0x75e:0x44 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 82
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 61
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 74                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x796:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2019                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x79c:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2034                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 9                                   // Abbrev [9] 0x7a2:0x40 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 78
; CHECK-NEXT: .b8 75
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 104
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 73
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 111                                 // DW_AT_name
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 38
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 74                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 2039                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 3                                   // DW_AT_accessibility
; CHECK-NEXT:                                         // DW_ACCESS_private
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0x7db:0x6 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2019                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_artificial
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x7e3:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 2024                               // DW_AT_type
; CHECK-NEXT: .b8 13                                  // Abbrev [13] 0x7e8:0x5 DW_TAG_const_type
; CHECK-NEXT: .b32 1456                               // DW_AT_type
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x7ed:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 1456                               // DW_AT_type
; CHECK-NEXT: .b8 14                                  // Abbrev [14] 0x7f2:0x5 DW_TAG_reference_type
; CHECK-NEXT: .b32 2024                               // DW_AT_type
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x7f7:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 1456                               // DW_AT_type
; CHECK-NEXT: .b8 15                                  // Abbrev [15] 0x7fc:0x6 DW_TAG_subprogram
; CHECK-NEXT: .b32 1487                               // DW_AT_specification
; CHECK-NEXT: .b8 1                                   // DW_AT_inline
; CHECK-NEXT: .b8 21                                  // Abbrev [21] 0x802:0x32 DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 80
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 114                                 // DW_AT_name
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 1                                   // DW_AT_inline
; CHECK-NEXT: .b8 22                                  // Abbrev [22] 0x816:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 120                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2100                               // DW_AT_type
; CHECK-NEXT: .b8 22                                  // Abbrev [22] 0x81f:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 121                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2100                               // DW_AT_type
; CHECK-NEXT: .b8 22                                  // Abbrev [22] 0x828:0xb DW_TAG_formal_parameter
; CHECK-NEXT: .b8 114                                 // DW_AT_name
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2109                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 10                                  // Abbrev [10] 0x834:0x9 DW_TAG_base_type
; CHECK-NEXT: .b8 102                                 // DW_AT_name
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 12                                  // Abbrev [12] 0x83d:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 2100                               // DW_AT_type
; CHECK-NEXT: .b8 23                                  // Abbrev [23] 0x842:0xd5 DW_TAG_subprogram
; CHECK-NEXT: .b64 $L__func_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__func_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 80
; CHECK-NEXT: .b8 102
; CHECK-NEXT: .b8 83
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 115                                 // DW_AT_name
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 121
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 24                                  // Abbrev [24] 0x86d:0x10 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 2                                   // DW_AT_address_class
; CHECK-NEXT: .b8 5                                   // DW_AT_location
; CHECK-NEXT: .b8 144
; CHECK-NEXT: .b8 178
; CHECK-NEXT: .b8 228
; CHECK-NEXT: .b8 149
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 110                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2384                               // DW_AT_type
; CHECK-NEXT: .b8 25                                  // Abbrev [25] 0x87d:0xd DW_TAG_formal_parameter
; CHECK-NEXT: .b32 $L__debug_loc0                     // DW_AT_location
; CHECK-NEXT: .b8 97                                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2100                               // DW_AT_type
; CHECK-NEXT: .b8 22                                  // Abbrev [22] 0x88a:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 120                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2109                               // DW_AT_type
; CHECK-NEXT: .b8 22                                  // Abbrev [22] 0x893:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 121                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 5                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2109                               // DW_AT_type
; CHECK-NEXT: .b8 26                                  // Abbrev [26] 0x89c:0xd DW_TAG_variable
; CHECK-NEXT: .b32 $L__debug_loc1                     // DW_AT_location
; CHECK-NEXT: .b8 105                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 2384                               // DW_AT_type
; CHECK-NEXT: .b8 27                                  // Abbrev [27] 0x8a9:0x18 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 691                                // DW_AT_abstract_origin
; CHECK-NEXT: .b64 $L__tmp1                           // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__tmp2                           // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_call_file
; CHECK-NEXT: .b8 6                                   // DW_AT_call_line
; CHECK-NEXT: .b8 11                                  // DW_AT_call_column
; CHECK-NEXT: .b8 27                                  // Abbrev [27] 0x8c1:0x18 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 1450                               // DW_AT_abstract_origin
; CHECK-NEXT: .b64 $L__tmp2                           // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__tmp3                           // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_call_file
; CHECK-NEXT: .b8 6                                   // DW_AT_call_line
; CHECK-NEXT: .b8 24                                  // DW_AT_call_column
; CHECK-NEXT: .b8 27                                  // Abbrev [27] 0x8d9:0x18 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 2044                               // DW_AT_abstract_origin
; CHECK-NEXT: .b64 $L__tmp3                           // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__tmp4                           // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_call_file
; CHECK-NEXT: .b8 6                                   // DW_AT_call_line
; CHECK-NEXT: .b8 37                                  // DW_AT_call_column
; CHECK-NEXT: .b8 28                                  // Abbrev [28] 0x8f1:0x25 DW_TAG_inlined_subroutine
; CHECK-NEXT: .b32 2050                               // DW_AT_abstract_origin
; CHECK-NEXT: .b64 $L__tmp9                           // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__tmp10                          // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_call_file
; CHECK-NEXT: .b8 8                                   // DW_AT_call_line
; CHECK-NEXT: .b8 5                                   // DW_AT_call_column
; CHECK-NEXT: .b8 29                                  // Abbrev [29] 0x909:0xc DW_TAG_formal_parameter
; CHECK-NEXT: .b8 2                                   // DW_AT_address_class
; CHECK-NEXT: .b8 5                                   // DW_AT_location
; CHECK-NEXT: .b8 144
; CHECK-NEXT: .b8 179
; CHECK-NEXT: .b8 204
; CHECK-NEXT: .b8 149
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b32 2079                               // DW_AT_abstract_origin
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 30                                  // Abbrev [30] 0x917:0xd DW_TAG_namespace
; CHECK-NEXT: .b8 115                                 // DW_AT_name
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 31                                  // Abbrev [31] 0x91c:0x7 DW_TAG_imported_declaration
; CHECK-NEXT: .b8 4                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 202                                 // DW_AT_decl_line
; CHECK-NEXT: .b32 2340                               // DW_AT_import
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 32                                  // Abbrev [32] 0x924:0x1b DW_TAG_subprogram
; CHECK-NEXT: .b8 95                                  // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 90
; CHECK-NEXT: .b8 76
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 120
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 97                                  // DW_AT_name
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 44                                  // DW_AT_decl_line
; CHECK-NEXT: .b32 2367                               // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_declaration
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0x939:0x5 DW_TAG_formal_parameter
; CHECK-NEXT: .b32 2367                               // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 10                                  // Abbrev [10] 0x93f:0x11 DW_TAG_base_type
; CHECK-NEXT: .b8 108                                 // DW_AT_name
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                   // DW_AT_encoding
; CHECK-NEXT: .b8 8                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 10                                  // Abbrev [10] 0x950:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105                                 // DW_AT_name
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: 	}
; CHECK-NEXT: 	.section	.debug_macinfo	{	}
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
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !7, line: 202)
!5 = !DINamespace(name: "std", scope: null)
!6 = !DISubprogram(name: "abs", linkageName: "_ZL3absx", scope: !7, file: !7, line: 44, type: !8, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true)
!7 = !DIFile(filename: "clang/include/__clang_cuda_math_forward_declares.h", directory: "/some/directory")
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!15 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!70 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!144 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!365 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
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
