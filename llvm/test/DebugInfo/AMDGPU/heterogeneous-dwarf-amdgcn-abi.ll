; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -verify-machineinstrs -filetype=obj -emit-heterogeneous-dwarf-as-user-ops=false < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefixes=CHECK,CHECK-ORIG-OPS %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -verify-machineinstrs -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefixes=CHECK,CHECK-USER-OPS %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

%struct.StructEmpty = type { i8 }
%struct.StructSingleElement = type { i8 }
%struct.StructSingleElementRecursive = type { %struct.StructSingleElement }
%struct.StructTrivialCopyTrivialMove = type { i8 }
%struct.StructNoCopyTrivialMove = type { i8 }
%struct.StructTrivialCopyNoMove = type { i8 }
%struct.StructNoCopyNoMove = type { i8 }
%struct.StructNBytes = type { i8, [1 x i8] }
%struct.StructNBytes.0 = type { i8, [2 x i8] }
%struct.StructNBytes.1 = type { i8, [3 x i8] }
%struct.StructNBytes.2 = type { i8, [4 x i8] }
%struct.StructNBytes.3 = type { i8, [5 x i8] }
%struct.StructNBytes.4 = type { i8, [6 x i8] }
%struct.StructNBytes.5 = type { i8, [7 x i8] }
%struct.StructNBytes.6 = type { i8, [8 x i8] }
%struct.StructNBytes.7 = type { i8, [63 x i8] }
%struct.StructSinglePointerElement = type { ptr }
%struct.StructPointerElements = type { ptr, ptr }
%struct.StructMultipleElements = type { i32, i64 }

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructEmpty")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z21Test_Func_StructEmpty11StructEmpty() #0 !dbg !24 {
entry:
  %tmp = alloca %struct.StructEmpty, align 1, addrspace(5)
  %0 = addrspacecast ptr addrspace(5) %tmp to ptr
  call void @llvm.dbg.def(metadata !30, metadata ptr addrspace(5) %tmp), !dbg !31
  ret void, !dbg !32
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #1

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructEmpty")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z21Test_Kern_StructEmpty11StructEmpty(ptr addrspace(4) byref(%struct.StructEmpty) align 1 %0) #2 !dbg !33 {
entry:
  %coerce = alloca %struct.StructEmpty, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 1, i1 false)
  call void @llvm.dbg.def(metadata !36, metadata ptr addrspace(5) %coerce), !dbg !37
  ret void, !dbg !38
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0.p4.i64(i8* noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #3

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructSingleElement")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z29Test_Func_StructSingleElement19StructSingleElement(i8 %.coerce) #0 !dbg !39 {
entry:
  %0 = alloca %struct.StructSingleElement, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructSingleElement, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !47, metadata ptr addrspace(5) %0), !dbg !48
  ret void, !dbg !49
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructSingleElement")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z29Test_Kern_StructSingleElement19StructSingleElement(i8 %.coerce) #2 !dbg !50 {
entry:
  %0 = alloca %struct.StructSingleElement, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructSingleElement, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !53, metadata ptr addrspace(5) %0), !dbg !54
  ret void, !dbg !55
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructSingleElementRecursive")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z38Test_Func_StructSingleElementRecursive28StructSingleElementRecursive(i8 %.coerce) #0 !dbg !56 {
entry:
  %0 = alloca %struct.StructSingleElementRecursive, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructSingleElementRecursive, ptr %1, i32 0, i32 0
  %coerce.dive1 = getelementptr inbounds %struct.StructSingleElement, ptr %coerce.dive, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive1, align 1
  call void @llvm.dbg.def(metadata !64, metadata ptr addrspace(5) %0), !dbg !65
  ret void, !dbg !66
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructSingleElementRecursive")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z38Test_Kern_StructSingleElementRecursive28StructSingleElementRecursive(i8 %.coerce) #2 !dbg !67 {
entry:
  %0 = alloca %struct.StructSingleElementRecursive, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructSingleElementRecursive, ptr %1, i32 0, i32 0
  %coerce.dive1 = getelementptr inbounds %struct.StructSingleElement, ptr %coerce.dive, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive1, align 1
  call void @llvm.dbg.def(metadata !70, metadata ptr addrspace(5) %0), !dbg !71
  ret void, !dbg !72
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructTrivialCopyTrivialMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z38Test_Func_StructTrivialCopyTrivialMove28StructTrivialCopyTrivialMove(i8 %.coerce) #0 !dbg !73 {
entry:
  %0 = alloca %struct.StructTrivialCopyTrivialMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructTrivialCopyTrivialMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !91, metadata ptr addrspace(5) %0), !dbg !92
  ret void, !dbg !93
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructTrivialCopyTrivialMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z38Test_Kern_StructTrivialCopyTrivialMove28StructTrivialCopyTrivialMove(i8 %.coerce) #2 !dbg !94 {
entry:
  %0 = alloca %struct.StructTrivialCopyTrivialMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructTrivialCopyTrivialMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !97, metadata ptr addrspace(5) %0), !dbg !98
  ret void, !dbg !99
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructNoCopyTrivialMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z33Test_Func_StructNoCopyTrivialMove23StructNoCopyTrivialMove(i8 %.coerce) #0 !dbg !100 {
entry:
  %0 = alloca %struct.StructNoCopyTrivialMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructNoCopyTrivialMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !118, metadata ptr addrspace(5) %0), !dbg !119
  ret void, !dbg !120
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructNoCopyTrivialMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z33Test_Kern_StructNoCopyTrivialMove23StructNoCopyTrivialMove(i8 %.coerce) #2 !dbg !121 {
entry:
  %0 = alloca %struct.StructNoCopyTrivialMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructNoCopyTrivialMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !124, metadata ptr addrspace(5) %0), !dbg !125
  ret void, !dbg !126
}

; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name ("Test_Func_StructTrivialCopyNoMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z33Test_Func_StructTrivialCopyNoMove23StructTrivialCopyNoMove(i8 %.coerce) #0 !dbg !127 {
entry:
  %0 = alloca %struct.StructTrivialCopyNoMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructTrivialCopyNoMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !145, metadata ptr addrspace(5) %0), !dbg !146
  ret void, !dbg !147
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructTrivialCopyNoMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z33Test_Kern_StructTrivialCopyNoMove23StructTrivialCopyNoMove(i8 %.coerce) #2 !dbg !148 {
entry:
  %0 = alloca %struct.StructTrivialCopyNoMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructTrivialCopyNoMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !151, metadata ptr addrspace(5) %0), !dbg !152
  ret void, !dbg !153
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructNoCopyNoMove")
; FIXME: An existing bug in DwarfUnit::constructSubprogramArguments leads to
; this formal parameter not appearing at all in the resulting Dwarf.
; CHECK-NOT: DW_TAG_formal_parameter

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z28Test_Func_StructNoCopyNoMove18StructNoCopyNoMove(ptr addrspace(5) noundef %0) #0 !dbg !154 {
entry:
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  call void @llvm.dbg.def(metadata !172, metadata ptr addrspace(5) %0), !dbg !173
  ret void, !dbg !174
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructNoCopyNoMove")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z28Test_Kern_StructNoCopyNoMove18StructNoCopyNoMove(i8 %.coerce) #2 !dbg !175 {
entry:
  %0 = alloca %struct.StructNoCopyNoMove, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructNoCopyNoMove, ptr %1, i32 0, i32 0
  store i8 %.coerce, i8* %coerce.dive, align 1
  call void @llvm.dbg.def(metadata !178, metadata ptr addrspace(5) %0), !dbg !179
  ret void, !dbg !180
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct2Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct2Bytes12StructNBytesILj2EE(i16 %.coerce) #0 !dbg !181 {
entry:
  %0 = alloca %struct.StructNBytes, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  store i16 %.coerce, i16* %1, align 1
  call void @llvm.dbg.def(metadata !195, metadata ptr addrspace(5) %0), !dbg !196
  ret void, !dbg !197
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct2Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct2Bytes12StructNBytesILj2EE(ptr addrspace(4) byref(%struct.StructNBytes) align 1 %0) #2 !dbg !198 {
entry:
  %coerce = alloca %struct.StructNBytes, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 2, i1 false)
  call void @llvm.dbg.def(metadata !201, metadata ptr addrspace(5) %coerce), !dbg !202
  ret void, !dbg !203
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct3Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct3Bytes12StructNBytesILj3EE(i32 %.coerce) #0 !dbg !204 {
entry:
  %0 = alloca %struct.StructNBytes.0, align 1, addrspace(5)
  %tmp.coerce = alloca i32, align 4, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %tmp.coerce.ascast = addrspacecast ptr addrspace(5) %tmp.coerce to ptr
  store i32 %.coerce, ptr %tmp.coerce.ascast, align 4
  call void @llvm.memcpy.p0.p0.i64(i8* align 1 %1, i8* align 4 %tmp.coerce.ascast, i64 3, i1 false)
  call void @llvm.dbg.def(metadata !218, metadata ptr addrspace(5) %0), !dbg !219
  ret void, !dbg !220
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #3

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct3Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct3Bytes12StructNBytesILj3EE(ptr addrspace(4) byref(%struct.StructNBytes.0) align 1 %0) #2 !dbg !221 {
entry:
  %coerce = alloca %struct.StructNBytes.0, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 3, i1 false)
  call void @llvm.dbg.def(metadata !224, metadata ptr addrspace(5) %coerce), !dbg !225
  ret void, !dbg !226
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct4Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct4Bytes12StructNBytesILj4EE(i32 %.coerce) #0 !dbg !227 {
entry:
  %0 = alloca %struct.StructNBytes.1, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  store i32 %.coerce, ptr %1, align 1
  call void @llvm.dbg.def(metadata !241, metadata ptr addrspace(5) %0), !dbg !242
  ret void, !dbg !243
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct4Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct4Bytes12StructNBytesILj4EE(ptr addrspace(4) byref(%struct.StructNBytes.1) align 1 %0) #2 !dbg !244 {
entry:
  %coerce = alloca %struct.StructNBytes.1, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 4, i1 false)
  call void @llvm.dbg.def(metadata !247, metadata ptr addrspace(5) %coerce), !dbg !248
  ret void, !dbg !249
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct5Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct5Bytes12StructNBytesILj5EE([2 x i32] %.coerce) #0 !dbg !250 {
entry:
  %0 = alloca %struct.StructNBytes.2, align 1, addrspace(5)
  %tmp.coerce = alloca [2 x i32], align 4, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %tmp.coerce.ascast = addrspacecast ptr addrspace(5) %tmp.coerce to ptr
  store [2 x i32] %.coerce, ptr %tmp.coerce.ascast, align 4
  call void @llvm.memcpy.p0.p0.i64(i8* align 1 %1, i8* align 4 %tmp.coerce.ascast, i64 5, i1 false)
  call void @llvm.dbg.def(metadata !264, metadata ptr addrspace(5) %0), !dbg !265
  ret void, !dbg !266
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct5Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct5Bytes12StructNBytesILj5EE(ptr addrspace(4) byref(%struct.StructNBytes.2) align 1 %0) #2 !dbg !267 {
entry:
  %coerce = alloca %struct.StructNBytes.2, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 5, i1 false)
  call void @llvm.dbg.def(metadata !270, metadata ptr addrspace(5) %coerce), !dbg !271
  ret void, !dbg !272
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct6Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct6Bytes12StructNBytesILj6EE([2 x i32] %.coerce) #0 !dbg !273 {
entry:
  %0 = alloca %struct.StructNBytes.3, align 1, addrspace(5)
  %tmp.coerce = alloca [2 x i32], align 4, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %tmp.coerce.ascast = addrspacecast ptr addrspace(5) %tmp.coerce to ptr
  store [2 x i32] %.coerce, ptr %tmp.coerce.ascast, align 4
  call void @llvm.memcpy.p0.p0.i64(i8* align 1 %1, i8* align 4 %tmp.coerce.ascast, i64 6, i1 false)
  call void @llvm.dbg.def(metadata !287, metadata ptr addrspace(5) %0), !dbg !288
  ret void, !dbg !289
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct6Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct6Bytes12StructNBytesILj6EE(ptr addrspace(4) byref(%struct.StructNBytes.3) align 1 %0) #2 !dbg !290 {
entry:
  %coerce = alloca %struct.StructNBytes.3, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 6, i1 false)
  call void @llvm.dbg.def(metadata !293, metadata ptr addrspace(5) %coerce), !dbg !294
  ret void, !dbg !295
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct7Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct7Bytes12StructNBytesILj7EE([2 x i32] %.coerce) #0 !dbg !296 {
entry:
  %0 = alloca %struct.StructNBytes.4, align 1, addrspace(5)
  %tmp.coerce = alloca [2 x i32], align 4, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %tmp.coerce.ascast = addrspacecast ptr addrspace(5) %tmp.coerce to ptr
  store [2 x i32] %.coerce, ptr %tmp.coerce.ascast, align 4
  call void @llvm.memcpy.p0.p0.i64(i8* align 1 %1, i8* align 4 %tmp.coerce.ascast, i64 7, i1 false)
  call void @llvm.dbg.def(metadata !310, metadata ptr addrspace(5) %0), !dbg !311
  ret void, !dbg !312
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct7Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct7Bytes12StructNBytesILj7EE(ptr addrspace(4) byref(%struct.StructNBytes.4) align 1 %0) #2 !dbg !313 {
entry:
  %coerce = alloca %struct.StructNBytes.4, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 7, i1 false)
  call void @llvm.dbg.def(metadata !316, metadata ptr addrspace(5) %coerce), !dbg !317
  ret void, !dbg !318
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct8Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct8Bytes12StructNBytesILj8EE([2 x i32] %.coerce) #0 !dbg !319 {
entry:
  %0 = alloca %struct.StructNBytes.5, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  store [2 x i32] %.coerce, ptr %1, align 1
  call void @llvm.dbg.def(metadata !333, metadata ptr addrspace(5) %0), !dbg !334
  ret void, !dbg !335
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct8Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct8Bytes12StructNBytesILj8EE(ptr addrspace(4) byref(%struct.StructNBytes.5) align 1 %0) #2 !dbg !336 {
entry:
  %coerce = alloca %struct.StructNBytes.5, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 8, i1 false)
  call void @llvm.dbg.def(metadata !339, metadata ptr addrspace(5) %coerce), !dbg !340
  ret void, !dbg !341
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct9Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z22Test_Func_Struct9Bytes12StructNBytesILj9EE(i8 %.coerce0, [8 x i8] %.coerce1) #0 !dbg !342 {
entry:
  %0 = alloca %struct.StructNBytes.6, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %2 = getelementptr inbounds %struct.StructNBytes.6, ptr %1, i32 0, i32 0
  store i8 %.coerce0, i8* %2, align 1
  %3 = getelementptr inbounds %struct.StructNBytes.6, ptr %1, i32 0, i32 1
  store [8 x i8] %.coerce1, [8 x i8]* %3, align 1
  call void @llvm.dbg.def(metadata !356, metadata ptr addrspace(5) %0), !dbg !357
  ret void, !dbg !358
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct9Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct9Bytes12StructNBytesILj9EE(ptr addrspace(4) byref(%struct.StructNBytes.6) align 1 %0) #2 !dbg !359 {
entry:
  %coerce = alloca %struct.StructNBytes.6, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 9, i1 false)
  call void @llvm.dbg.def(metadata !362, metadata ptr addrspace(5) %coerce), !dbg !363
  ret void, !dbg !364
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Struct64Bytes")
; CHECK: DW_TAG_formal_parameter
; FIXME: fix byval
; CHECK: DW_AT_location (<empty>)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z23Test_Func_Struct64Bytes12StructNBytesILj64EE(ptr addrspace(5) noundef byval(%struct.StructNBytes.7) align 1 %0) #0 !dbg !365 {
entry:
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  call void @llvm.dbg.def(metadata !379, metadata ptr addrspace(5) %0), !dbg !380
  ret void, !dbg !381
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Struct64Bytes")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z23Test_Kern_Struct64Bytes12StructNBytesILj64EE(ptr addrspace(4) byref(%struct.StructNBytes.7) align 1 %0) #2 !dbg !382 {
entry:
  %coerce = alloca %struct.StructNBytes.7, align 1, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 1 %1, ptr addrspace(4) align 1 %0, i64 64, i1 false)
  call void @llvm.dbg.def(metadata !385, metadata ptr addrspace(5) %coerce), !dbg !386
  ret void, !dbg !387
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Int8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z15Test_Func_Int8Tc(i8 noundef signext %0) #0 !dbg !388 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !393, metadata ptr addrspace(5) %.addr), !dbg !394
  ret void, !dbg !395
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Int8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z15Test_Kern_Int8Tc(i8 noundef %0) #2 !dbg !396 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !399, metadata ptr addrspace(5) %.addr), !dbg !400
  ret void, !dbg !401
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_UInt8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z16Test_Func_UInt8Th(i8 noundef zeroext %0) #0 !dbg !402 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !407, metadata ptr addrspace(5) %.addr), !dbg !408
  ret void, !dbg !409
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_UInt8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z16Test_Kern_UInt8Th(i8 noundef %0) #2 !dbg !410 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !413, metadata ptr addrspace(5) %.addr), !dbg !414
  ret void, !dbg !415
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Int16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z16Test_Func_Int16Ts(i16 noundef signext %0) #0 !dbg !416 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !421, metadata ptr addrspace(5) %.addr), !dbg !422
  ret void, !dbg !423
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Int16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z16Test_Kern_Int16Ts(i16 noundef %0) #2 !dbg !424 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !427, metadata ptr addrspace(5) %.addr), !dbg !428
  ret void, !dbg !429
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_UInt16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z17Test_Func_UInt16Tt(i16 noundef zeroext %0) #0 !dbg !430 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !435, metadata ptr addrspace(5) %.addr), !dbg !436
  ret void, !dbg !437
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_UInt16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z17Test_Kern_UInt16Tt(i16 noundef %0) #2 !dbg !438 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !441, metadata ptr addrspace(5) %.addr), !dbg !442
  ret void, !dbg !443
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Int32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z16Test_Func_Int32Ti(i32 noundef %0) #0 !dbg !444 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !449, metadata ptr addrspace(5) %.addr), !dbg !450
  ret void, !dbg !451
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Int32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z16Test_Kern_Int32Ti(i32 noundef %0) #2 !dbg !452 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !455, metadata ptr addrspace(5) %.addr), !dbg !456
  ret void, !dbg !457
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_UInt32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z17Test_Func_UInt32Tj(i32 noundef %0) #0 !dbg !458 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !463, metadata ptr addrspace(5) %.addr), !dbg !464
  ret void, !dbg !465
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_UInt32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z17Test_Kern_UInt32Tj(i32 noundef %0) #2 !dbg !466 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !469, metadata ptr addrspace(5) %.addr), !dbg !470
  ret void, !dbg !471
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Int64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z16Test_Func_Int64Tl(i64 noundef %0) #0 !dbg !472 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !477, metadata ptr addrspace(5) %.addr), !dbg !478
  ret void, !dbg !479
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Int64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z16Test_Kern_Int64Tl(i64 noundef %0) #2 !dbg !480 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !483, metadata ptr addrspace(5) %.addr), !dbg !484
  ret void, !dbg !485
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_UInt64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z17Test_Func_UInt64Tm(i64 noundef %0) #0 !dbg !486 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !491, metadata ptr addrspace(5) %.addr), !dbg !492
  ret void, !dbg !493
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_UInt64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z17Test_Kern_UInt64Tm(i64 noundef %0) #2 !dbg !494 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !497, metadata ptr addrspace(5) %.addr), !dbg !498
  ret void, !dbg !499
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumInt8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z19Test_Func_EnumInt8T9EnumInt8T(i8 noundef signext %0) #0 !dbg !500 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !505, metadata ptr addrspace(5) %.addr), !dbg !506
  ret void, !dbg !507
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumInt8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z19Test_Kern_EnumInt8T9EnumInt8T(i8 noundef %0) #2 !dbg !508 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !511, metadata ptr addrspace(5) %.addr), !dbg !512
  ret void, !dbg !513
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumUInt8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z20Test_Func_EnumUInt8T10EnumUInt8T(i8 noundef zeroext %0) #0 !dbg !514 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !519, metadata ptr addrspace(5) %.addr), !dbg !520
  ret void, !dbg !521
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumUInt8T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumUInt8T10EnumUInt8T(i8 noundef %0) #2 !dbg !522 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  store i8 %0, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !525, metadata ptr addrspace(5) %.addr), !dbg !526
  ret void, !dbg !527
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumInt16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z20Test_Func_EnumInt16T10EnumInt16T(i16 noundef signext %0) #0 !dbg !528 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !533, metadata ptr addrspace(5) %.addr), !dbg !534
  ret void, !dbg !535
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumInt16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumInt16T10EnumInt16T(i16 noundef %0) #2 !dbg !536 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !539, metadata ptr addrspace(5) %.addr), !dbg !540
  ret void, !dbg !541
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumUInt16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z21Test_Func_EnumUInt16T11EnumUInt16T(i16 noundef zeroext %0) #0 !dbg !542 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !547, metadata ptr addrspace(5) %.addr), !dbg !548
  ret void, !dbg !549
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumUInt16T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z21Test_Kern_EnumUInt16T11EnumUInt16T(i16 noundef %0) #2 !dbg !550 {
entry:
  %.addr = alloca i16, align 2, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i16*
  store i16 %0, i16* %.addr.ascast, align 2
  call void @llvm.dbg.def(metadata !553, metadata ptr addrspace(5) %.addr), !dbg !554
  ret void, !dbg !555
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumInt32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z20Test_Func_EnumInt32T10EnumInt32T(i32 noundef %0) #0 !dbg !556 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !561, metadata ptr addrspace(5) %.addr), !dbg !562
  ret void, !dbg !563
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumInt32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumInt32T10EnumInt32T(i32 noundef %0) #2 !dbg !564 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !567, metadata ptr addrspace(5) %.addr), !dbg !568
  ret void, !dbg !569
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumUInt32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z21Test_Func_EnumUInt32T11EnumUInt32T(i32 noundef %0) #0 !dbg !570 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !575, metadata ptr addrspace(5) %.addr), !dbg !576
  ret void, !dbg !577
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumUInt32T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z21Test_Kern_EnumUInt32T11EnumUInt32T(i32 noundef %0) #2 !dbg !578 {
entry:
  %.addr = alloca i32, align 4, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store i32 %0, ptr %.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !581, metadata ptr addrspace(5) %.addr), !dbg !582
  ret void, !dbg !583
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumInt64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z20Test_Func_EnumInt64T10EnumInt64T(i64 noundef %0) #0 !dbg !584 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !589, metadata ptr addrspace(5) %.addr), !dbg !590
  ret void, !dbg !591
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumInt64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumInt64T10EnumInt64T(i64 noundef %0) #2 !dbg !592 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !595, metadata ptr addrspace(5) %.addr), !dbg !596
  ret void, !dbg !597
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_EnumUInt64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z21Test_Func_EnumUInt64T11EnumUInt64T(i64 noundef %0) #0 !dbg !598 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !603, metadata ptr addrspace(5) %.addr), !dbg !604
  ret void, !dbg !605
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_EnumUInt64T")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z21Test_Kern_EnumUInt64T11EnumUInt64T(i64 noundef %0) #2 !dbg !606 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !609, metadata ptr addrspace(5) %.addr), !dbg !610
  ret void, !dbg !611
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_PromotableInteger")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z27Test_Func_PromotableIntegerb(i1 noundef zeroext %0) #0 !dbg !612 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  %frombool = zext i1 %0 to i8
  store i8 %frombool, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !618, metadata ptr addrspace(5) %.addr), !dbg !619
  ret void, !dbg !620
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_PromotableInteger")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit4, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z27Test_Kern_PromotableIntegerb(i1 noundef %0) #2 !dbg !621 {
entry:
  %.addr = alloca i8, align 1, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i8*
  %frombool = zext i1 %0 to i8
  store i8 %frombool, i8* %.addr.ascast, align 1
  call void @llvm.dbg.def(metadata !624, metadata ptr addrspace(5) %.addr), !dbg !625
  ret void, !dbg !626
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Pointer")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z17Test_Func_PointerPi(ptr noundef %0) #0 !dbg !627 {
entry:
  %.addr = alloca ptr, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store ptr %0, ptr %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !633, metadata ptr addrspace(5) %.addr), !dbg !634
  ret void, !dbg !635
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Pointer")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z17Test_Kern_PointerPi(ptr addrspace(1) noundef %.coerce) #2 !dbg !636 {
entry:
  %0 = alloca ptr, align 8, addrspace(5)
  %.addr = alloca ptr, align 8, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  %2 = addrspacecast ptr addrspace(1) %.coerce to ptr
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  store ptr %3, ptr %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !639, metadata ptr addrspace(5) %.addr), !dbg !640
  ret void, !dbg !641
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_Reference")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z19Test_Func_ReferenceRi(ptr noundef nonnull align 4 dereferenceable(4) %0) #0 !dbg !642 {
entry:
  %.addr = alloca ptr, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  store ptr %0, ptr %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !648, metadata ptr addrspace(5) %.addr), !dbg !649
  ret void, !dbg !650
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_Reference")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z19Test_Kern_ReferenceRi(ptr addrspace(1) noundef nonnull align 4 dereferenceable(4) %.coerce) #2 !dbg !651 {
entry:
  %0 = alloca ptr, align 8, addrspace(5)
  %.addr = alloca ptr, align 8, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to ptr
  %2 = addrspacecast ptr addrspace(1) %.coerce to ptr
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  store ptr %3, ptr %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !654, metadata ptr addrspace(5) %.addr), !dbg !655
  ret void, !dbg !656
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructSinglePointerElement")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z36Test_Func_StructSinglePointerElement26StructSinglePointerElement(ptr %.coerce) #0 !dbg !657 {
entry:
  %0 = alloca %struct.StructSinglePointerElement, align 8, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructSinglePointerElement, ptr %1, i32 0, i32 0
  store ptr %.coerce, ptr %coerce.dive, align 8
  call void @llvm.dbg.def(metadata !665, metadata ptr addrspace(5) %0), !dbg !666
  ret void, !dbg !667
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructSinglePointerElement")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z36Test_Kern_StructSinglePointerElement26StructSinglePointerElement(ptr addrspace(1) %.coerce) #2 !dbg !668 {
entry:
  %0 = alloca %struct.StructSinglePointerElement, align 8, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %coerce.dive = getelementptr inbounds %struct.StructSinglePointerElement, ptr %1, i32 0, i32 0
  %2 = addrspacecast ptr addrspace(1) %.coerce to ptr
  store ptr %2, ptr %coerce.dive, align 8
  call void @llvm.dbg.def(metadata !671, metadata ptr addrspace(5) %0), !dbg !672
  ret void, !dbg !673
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_StructPointerElements")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z31Test_Func_StructPointerElements21StructPointerElements(ptr %.coerce0, ptr %.coerce1) #0 !dbg !674 {
entry:
  %0 = alloca %struct.StructPointerElements, align 8, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %0 to ptr
  %2 = getelementptr inbounds %struct.StructPointerElements, ptr %1, i32 0, i32 0
  store ptr %.coerce0, ptr %2, align 8
  %3 = getelementptr inbounds %struct.StructPointerElements, ptr %1, i32 0, i32 1
  store ptr %.coerce1, float** %3, align 8
  call void @llvm.dbg.def(metadata !685, metadata ptr addrspace(5) %0), !dbg !686
  ret void, !dbg !687
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_StructPointerElements")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z31Test_Kern_StructPointerElements21StructPointerElements(ptr addrspace(4) byref(%struct.StructPointerElements) align 8 %0) #2 !dbg !688 {
entry:
  %coerce = alloca %struct.StructPointerElements, align 8, addrspace(5)
  %1 = addrspacecast ptr addrspace(5) %coerce to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 8 %1, ptr addrspace(4) align 8 %0, i64 16, i1 false)
  call void @llvm.dbg.def(metadata !691, metadata ptr addrspace(5) %coerce), !dbg !692
  ret void, !dbg !693
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_ParamRegLimitExpandedStruct")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit0, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z37Test_Func_ParamRegLimitExpandedStructlllllli22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i32 noundef %6, i32 %.coerce0, i64 %.coerce1) #0 !dbg !694 {
entry:
  %7 = alloca %struct.StructMultipleElements, align 8, addrspace(5)
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr1 = alloca i64, align 8, addrspace(5)
  %.addr2 = alloca i64, align 8, addrspace(5)
  %.addr3 = alloca i64, align 8, addrspace(5)
  %.addr4 = alloca i64, align 8, addrspace(5)
  %.addr5 = alloca i64, align 8, addrspace(5)
  %.addr6 = alloca i32, align 4, addrspace(5)
  %8 = addrspacecast ptr addrspace(5) %7 to ptr
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  %.addr1.ascast = addrspacecast ptr addrspace(5) %.addr1 to i64*
  %.addr2.ascast = addrspacecast ptr addrspace(5) %.addr2 to i64*
  %.addr3.ascast = addrspacecast ptr addrspace(5) %.addr3 to i64*
  %.addr4.ascast = addrspacecast ptr addrspace(5) %.addr4 to i64*
  %.addr5.ascast = addrspacecast ptr addrspace(5) %.addr5 to i64*
  %.addr6.ascast = addrspacecast ptr addrspace(5) %.addr6 to ptr
  %9 = getelementptr inbounds %struct.StructMultipleElements, ptr %8, i32 0, i32 0
  store i32 %.coerce0, ptr %9, align 8
  %10 = getelementptr inbounds %struct.StructMultipleElements, ptr %8, i32 0, i32 1
  store i64 %.coerce1, i64* %10, align 8
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !710, metadata ptr addrspace(5) %.addr), !dbg !711
  store i64 %1, i64* %.addr1.ascast, align 8
  call void @llvm.dbg.def(metadata !712, metadata ptr addrspace(5) %.addr1), !dbg !713
  store i64 %2, i64* %.addr2.ascast, align 8
  call void @llvm.dbg.def(metadata !714, metadata ptr addrspace(5) %.addr2), !dbg !715
  store i64 %3, i64* %.addr3.ascast, align 8
  call void @llvm.dbg.def(metadata !716, metadata ptr addrspace(5) %.addr3), !dbg !717
  store i64 %4, i64* %.addr4.ascast, align 8
  call void @llvm.dbg.def(metadata !718, metadata ptr addrspace(5) %.addr4), !dbg !719
  store i64 %5, i64* %.addr5.ascast, align 8
  call void @llvm.dbg.def(metadata !720, metadata ptr addrspace(5) %.addr5), !dbg !721
  store i32 %6, ptr %.addr6.ascast, align 4
  call void @llvm.dbg.def(metadata !722, metadata ptr addrspace(5) %.addr6), !dbg !723
  call void @llvm.dbg.def(metadata !724, metadata ptr addrspace(5) %7), !dbg !725
  ret void, !dbg !726
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_ParamRegLimitExpandedStruct")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x48, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x48, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z37Test_Kern_ParamRegLimitExpandedStructlllllli22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i32 noundef %6, ptr addrspace(4) byref(%struct.StructMultipleElements) align 8 %7) #2 !dbg !727 {
entry:
  %coerce = alloca %struct.StructMultipleElements, align 8, addrspace(5)
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr1 = alloca i64, align 8, addrspace(5)
  %.addr2 = alloca i64, align 8, addrspace(5)
  %.addr3 = alloca i64, align 8, addrspace(5)
  %.addr4 = alloca i64, align 8, addrspace(5)
  %.addr5 = alloca i64, align 8, addrspace(5)
  %.addr6 = alloca i32, align 4, addrspace(5)
  %8 = addrspacecast ptr addrspace(5) %coerce to ptr
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  %.addr1.ascast = addrspacecast ptr addrspace(5) %.addr1 to i64*
  %.addr2.ascast = addrspacecast ptr addrspace(5) %.addr2 to i64*
  %.addr3.ascast = addrspacecast ptr addrspace(5) %.addr3 to i64*
  %.addr4.ascast = addrspacecast ptr addrspace(5) %.addr4 to i64*
  %.addr5.ascast = addrspacecast ptr addrspace(5) %.addr5 to i64*
  %.addr6.ascast = addrspacecast ptr addrspace(5) %.addr6 to ptr
  call void @llvm.memcpy.p0.p4.i64(i8* align 8 %8, ptr addrspace(4) align 8 %7, i64 16, i1 false)
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !737, metadata ptr addrspace(5) %.addr), !dbg !738
  store i64 %1, i64* %.addr1.ascast, align 8
  call void @llvm.dbg.def(metadata !739, metadata ptr addrspace(5) %.addr1), !dbg !740
  store i64 %2, i64* %.addr2.ascast, align 8
  call void @llvm.dbg.def(metadata !741, metadata ptr addrspace(5) %.addr2), !dbg !742
  store i64 %3, i64* %.addr3.ascast, align 8
  call void @llvm.dbg.def(metadata !743, metadata ptr addrspace(5) %.addr3), !dbg !744
  store i64 %4, i64* %.addr4.ascast, align 8
  call void @llvm.dbg.def(metadata !745, metadata ptr addrspace(5) %.addr4), !dbg !746
  store i64 %5, i64* %.addr5.ascast, align 8
  call void @llvm.dbg.def(metadata !747, metadata ptr addrspace(5) %.addr5), !dbg !748
  store i32 %6, ptr %.addr6.ascast, align 4
  call void @llvm.dbg.def(metadata !749, metadata ptr addrspace(5) %.addr6), !dbg !750
  call void @llvm.dbg.def(metadata !751, metadata ptr addrspace(5) %coerce), !dbg !752
  ret void, !dbg !753
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Func_ParamRegLimitUnexpandedStruct")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_regx SGPR32_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; FIXME: fix byval
; CHECK: DW_AT_location (<empty>)

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z39Test_Func_ParamRegLimitUnexpandedStructlllllll22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, ptr addrspace(5) noundef byval(%struct.StructMultipleElements) align 8 %7) #0 !dbg !754 {
entry:
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr1 = alloca i64, align 8, addrspace(5)
  %.addr2 = alloca i64, align 8, addrspace(5)
  %.addr3 = alloca i64, align 8, addrspace(5)
  %.addr4 = alloca i64, align 8, addrspace(5)
  %.addr5 = alloca i64, align 8, addrspace(5)
  %.addr6 = alloca i64, align 8, addrspace(5)
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  %.addr1.ascast = addrspacecast ptr addrspace(5) %.addr1 to i64*
  %.addr2.ascast = addrspacecast ptr addrspace(5) %.addr2 to i64*
  %.addr3.ascast = addrspacecast ptr addrspace(5) %.addr3 to i64*
  %.addr4.ascast = addrspacecast ptr addrspace(5) %.addr4 to i64*
  %.addr5.ascast = addrspacecast ptr addrspace(5) %.addr5 to i64*
  %.addr6.ascast = addrspacecast ptr addrspace(5) %.addr6 to i64*
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !766, metadata ptr addrspace(5) %.addr), !dbg !767
  store i64 %1, i64* %.addr1.ascast, align 8
  call void @llvm.dbg.def(metadata !768, metadata ptr addrspace(5) %.addr1), !dbg !769
  store i64 %2, i64* %.addr2.ascast, align 8
  call void @llvm.dbg.def(metadata !770, metadata ptr addrspace(5) %.addr2), !dbg !771
  store i64 %3, i64* %.addr3.ascast, align 8
  call void @llvm.dbg.def(metadata !772, metadata ptr addrspace(5) %.addr3), !dbg !773
  store i64 %4, i64* %.addr4.ascast, align 8
  call void @llvm.dbg.def(metadata !774, metadata ptr addrspace(5) %.addr4), !dbg !775
  store i64 %5, i64* %.addr5.ascast, align 8
  call void @llvm.dbg.def(metadata !776, metadata ptr addrspace(5) %.addr5), !dbg !777
  store i64 %6, i64* %.addr6.ascast, align 8
  call void @llvm.dbg.def(metadata !778, metadata ptr addrspace(5) %.addr6), !dbg !779
  %8 = addrspacecast ptr addrspace(5) %7 to ptr
  call void @llvm.dbg.def(metadata !780, metadata ptr addrspace(5) %7), !dbg !781
  ret void, !dbg !782
}

; CHECK: DW_TAG_subprogram
; CHECK-LABEL: DW_AT_name ("Test_Kern_ParamRegLimitUnexpandedStruct")
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit24, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x28, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x30, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x38, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x40, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x48, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_constu 0x48, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK: DW_TAG_formal_parameter
; CHECK-ORIG-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_offset)
; CHECK-USER-OPS: DW_AT_location (DW_OP_lit0, DW_OP_stack_value, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @_Z39Test_Kern_ParamRegLimitUnexpandedStructlllllll22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, ptr addrspace(4) byref(%struct.StructMultipleElements) align 8 %7) #2 !dbg !783 {
entry:
  %coerce = alloca %struct.StructMultipleElements, align 8, addrspace(5)
  %.addr = alloca i64, align 8, addrspace(5)
  %.addr1 = alloca i64, align 8, addrspace(5)
  %.addr2 = alloca i64, align 8, addrspace(5)
  %.addr3 = alloca i64, align 8, addrspace(5)
  %.addr4 = alloca i64, align 8, addrspace(5)
  %.addr5 = alloca i64, align 8, addrspace(5)
  %.addr6 = alloca i64, align 8, addrspace(5)
  %8 = addrspacecast ptr addrspace(5) %coerce to ptr
  %.addr.ascast = addrspacecast ptr addrspace(5) %.addr to i64*
  %.addr1.ascast = addrspacecast ptr addrspace(5) %.addr1 to i64*
  %.addr2.ascast = addrspacecast ptr addrspace(5) %.addr2 to i64*
  %.addr3.ascast = addrspacecast ptr addrspace(5) %.addr3 to i64*
  %.addr4.ascast = addrspacecast ptr addrspace(5) %.addr4 to i64*
  %.addr5.ascast = addrspacecast ptr addrspace(5) %.addr5 to i64*
  %.addr6.ascast = addrspacecast ptr addrspace(5) %.addr6 to i64*
  call void @llvm.memcpy.p0.p4.i64(i8* align 8 %8, ptr addrspace(4) align 8 %7, i64 16, i1 false)
  store i64 %0, i64* %.addr.ascast, align 8
  call void @llvm.dbg.def(metadata !793, metadata ptr addrspace(5) %.addr), !dbg !794
  store i64 %1, i64* %.addr1.ascast, align 8
  call void @llvm.dbg.def(metadata !795, metadata ptr addrspace(5) %.addr1), !dbg !796
  store i64 %2, i64* %.addr2.ascast, align 8
  call void @llvm.dbg.def(metadata !797, metadata ptr addrspace(5) %.addr2), !dbg !798
  store i64 %3, i64* %.addr3.ascast, align 8
  call void @llvm.dbg.def(metadata !799, metadata ptr addrspace(5) %.addr3), !dbg !800
  store i64 %4, i64* %.addr4.ascast, align 8
  call void @llvm.dbg.def(metadata !801, metadata ptr addrspace(5) %.addr4), !dbg !802
  store i64 %5, i64* %.addr5.ascast, align 8
  call void @llvm.dbg.def(metadata !803, metadata ptr addrspace(5) %.addr5), !dbg !804
  store i64 %6, i64* %.addr6.ascast, align 8
  call void @llvm.dbg.def(metadata !805, metadata ptr addrspace(5) %.addr6), !dbg !806
  call void @llvm.dbg.def(metadata !807, metadata ptr addrspace(5) %coerce), !dbg !808
  ret void, !dbg !809
}

attributes #0 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,1024" "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #3 = { argmemonly nofree nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 4cf32d2891c002b7c81d8ed164c12c074b621388)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "clang/test/CodeGenHIP/<stdin>", directory: "/home/slinder1/llvm-project/amd-stg-open")
!2 = !{!3, !7, !9, !11, !13, !15, !17, !19}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumInt8T", file: !4, line: 80, baseType: !5, size: 8, elements: !6, identifier: "_ZTS9EnumInt8T")
!4 = !DIFile(filename: "clang/test/CodeGenHIP/debug-info-amdgcn-abi-heterogeneous-dwarf.hip", directory: "/home/slinder1/llvm-project/amd-stg-open")
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{}
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumUInt8T", file: !4, line: 81, baseType: !8, size: 8, elements: !6, identifier: "_ZTS10EnumUInt8T")
!8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumInt16T", file: !4, line: 82, baseType: !10, size: 16, elements: !6, identifier: "_ZTS10EnumInt16T")
!10 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!11 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumUInt16T", file: !4, line: 83, baseType: !12, size: 16, elements: !6, identifier: "_ZTS11EnumUInt16T")
!12 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumInt32T", file: !4, line: 84, baseType: !14, size: 32, elements: !6, identifier: "_ZTS10EnumInt32T")
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumUInt32T", file: !4, line: 85, baseType: !16, size: 32, elements: !6, identifier: "_ZTS11EnumUInt32T")
!16 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!17 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumInt64T", file: !4, line: 86, baseType: !18, size: 64, elements: !6, identifier: "_ZTS10EnumInt64T")
!18 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!19 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumUInt64T", file: !4, line: 87, baseType: !20, size: 64, elements: !6, identifier: "_ZTS11EnumUInt64T")
!20 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!21 = !{i32 2, !"Debug Info Version", i32 4}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 4cf32d2891c002b7c81d8ed164c12c074b621388)"}
!24 = distinct !DISubprogram(name: "Test_Func_StructEmpty", linkageName: "_Z21Test_Func_StructEmpty11StructEmpty", scope: !4, file: !4, line: 106, type: !25, scopeLine: 106, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !27}
!27 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructEmpty", file: !4, line: 32, size: 8, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTS11StructEmpty")
!28 = !{!29}
!29 = !DILocalVariable(arg: 1, scope: !24, file: !4, line: 106, type: !27)
!30 = distinct !DILifetime(object: !29, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructEmpty)))
!31 = !DILocation(line: 106, column: 50, scope: !24)
!32 = !DILocation(line: 106, column: 53, scope: !24)
!33 = distinct !DISubprogram(name: "Test_Kern_StructEmpty", linkageName: "_Z21Test_Kern_StructEmpty11StructEmpty", scope: !4, file: !4, line: 111, type: !25, scopeLine: 111, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
!34 = !{!35}
!35 = !DILocalVariable(arg: 1, scope: !33, file: !4, line: 111, type: !27)
!36 = distinct !DILifetime(object: !35, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructEmpty)))
!37 = !DILocation(line: 111, column: 50, scope: !33)
!38 = !DILocation(line: 111, column: 53, scope: !33)
!39 = distinct !DISubprogram(name: "Test_Func_StructSingleElement", linkageName: "_Z29Test_Func_StructSingleElement19StructSingleElement", scope: !4, file: !4, line: 117, type: !40, scopeLine: 117, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !42}
!42 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructSingleElement", file: !4, line: 34, size: 8, flags: DIFlagTypePassByValue, elements: !43, identifier: "_ZTS19StructSingleElement")
!43 = !{!44}
!44 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !42, file: !4, line: 35, baseType: !5, size: 8)
!45 = !{!46}
!46 = !DILocalVariable(arg: 1, scope: !39, file: !4, line: 117, type: !42)
!47 = distinct !DILifetime(object: !46, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructSingleElement)))
!48 = !DILocation(line: 117, column: 66, scope: !39)
!49 = !DILocation(line: 117, column: 69, scope: !39)
!50 = distinct !DISubprogram(name: "Test_Kern_StructSingleElement", linkageName: "_Z29Test_Kern_StructSingleElement19StructSingleElement", scope: !4, file: !4, line: 123, type: !40, scopeLine: 123, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !51)
!51 = !{!52}
!52 = !DILocalVariable(arg: 1, scope: !50, file: !4, line: 123, type: !42)
!53 = distinct !DILifetime(object: !52, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructSingleElement)))
!54 = !DILocation(line: 123, column: 66, scope: !50)
!55 = !DILocation(line: 123, column: 69, scope: !50)
!56 = distinct !DISubprogram(name: "Test_Func_StructSingleElementRecursive", linkageName: "_Z38Test_Func_StructSingleElementRecursive28StructSingleElementRecursive", scope: !4, file: !4, line: 129, type: !57, scopeLine: 129, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
!57 = !DISubroutineType(types: !58)
!58 = !{null, !59}
!59 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructSingleElementRecursive", file: !4, line: 38, size: 8, flags: DIFlagTypePassByValue, elements: !60, identifier: "_ZTS28StructSingleElementRecursive")
!60 = !{!61}
!61 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !59, file: !4, line: 39, baseType: !42, size: 8)
!62 = !{!63}
!63 = !DILocalVariable(arg: 1, scope: !56, file: !4, line: 129, type: !59)
!64 = distinct !DILifetime(object: !63, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructSingleElementRecursive)))
!65 = !DILocation(line: 129, column: 84, scope: !56)
!66 = !DILocation(line: 129, column: 87, scope: !56)
!67 = distinct !DISubprogram(name: "Test_Kern_StructSingleElementRecursive", linkageName: "_Z38Test_Kern_StructSingleElementRecursive28StructSingleElementRecursive", scope: !4, file: !4, line: 135, type: !57, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !68)
!68 = !{!69}
!69 = !DILocalVariable(arg: 1, scope: !67, file: !4, line: 135, type: !59)
!70 = distinct !DILifetime(object: !69, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructSingleElementRecursive)))
!71 = !DILocation(line: 135, column: 84, scope: !67)
!72 = !DILocation(line: 135, column: 87, scope: !67)
!73 = distinct !DISubprogram(name: "Test_Func_StructTrivialCopyTrivialMove", linkageName: "_Z38Test_Func_StructTrivialCopyTrivialMove28StructTrivialCopyTrivialMove", scope: !4, file: !4, line: 141, type: !74, scopeLine: 141, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !89)
!74 = !DISubroutineType(types: !75)
!75 = !{null, !76}
!76 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructTrivialCopyTrivialMove", file: !4, line: 42, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !77, identifier: "_ZTS28StructTrivialCopyTrivialMove")
!77 = !{!78, !79, !85}
!78 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !76, file: !4, line: 43, baseType: !5, size: 8)
!79 = !DISubprogram(name: "StructTrivialCopyTrivialMove", scope: !76, file: !4, line: 44, type: !80, scopeLine: 44, flags: DIFlagPrototyped, spFlags: 0)
!80 = !DISubroutineType(types: !81)
!81 = !{null, !82, !83}
!82 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !76, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!83 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !84, size: 64)
!84 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !76)
!85 = !DISubprogram(name: "StructTrivialCopyTrivialMove", scope: !76, file: !4, line: 45, type: !86, scopeLine: 45, flags: DIFlagPrototyped, spFlags: 0)
!86 = !DISubroutineType(types: !87)
!87 = !{null, !82, !88}
!88 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !76, size: 64)
!89 = !{!90}
!90 = !DILocalVariable(arg: 1, scope: !73, file: !4, line: 141, type: !76)
!91 = distinct !DILifetime(object: !90, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructTrivialCopyTrivialMove)))
!92 = !DILocation(line: 141, column: 84, scope: !73)
!93 = !DILocation(line: 141, column: 87, scope: !73)
!94 = distinct !DISubprogram(name: "Test_Kern_StructTrivialCopyTrivialMove", linkageName: "_Z38Test_Kern_StructTrivialCopyTrivialMove28StructTrivialCopyTrivialMove", scope: !4, file: !4, line: 147, type: !74, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !95)
!95 = !{!96}
!96 = !DILocalVariable(arg: 1, scope: !94, file: !4, line: 147, type: !76)
!97 = distinct !DILifetime(object: !96, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructTrivialCopyTrivialMove)))
!98 = !DILocation(line: 147, column: 84, scope: !94)
!99 = !DILocation(line: 147, column: 87, scope: !94)
!100 = distinct !DISubprogram(name: "Test_Func_StructNoCopyTrivialMove", linkageName: "_Z33Test_Func_StructNoCopyTrivialMove23StructNoCopyTrivialMove", scope: !4, file: !4, line: 153, type: !101, scopeLine: 153, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !116)
!101 = !DISubroutineType(types: !102)
!102 = !{null, !103}
!103 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNoCopyTrivialMove", file: !4, line: 48, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !104, identifier: "_ZTS23StructNoCopyTrivialMove")
!104 = !{!105, !106, !112}
!105 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !103, file: !4, line: 49, baseType: !5, size: 8)
!106 = !DISubprogram(name: "StructNoCopyTrivialMove", scope: !103, file: !4, line: 50, type: !107, scopeLine: 50, flags: DIFlagPrototyped, spFlags: DISPFlagDeleted)
!107 = !DISubroutineType(types: !108)
!108 = !{null, !109, !110}
!109 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !103, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!110 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !111, size: 64)
!111 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !103)
!112 = !DISubprogram(name: "StructNoCopyTrivialMove", scope: !103, file: !4, line: 51, type: !113, scopeLine: 51, flags: DIFlagPrototyped, spFlags: 0)
!113 = !DISubroutineType(types: !114)
!114 = !{null, !109, !115}
!115 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !103, size: 64)
!116 = !{!117}
!117 = !DILocalVariable(arg: 1, scope: !100, file: !4, line: 153, type: !103)
!118 = distinct !DILifetime(object: !117, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNoCopyTrivialMove)))
!119 = !DILocation(line: 153, column: 74, scope: !100)
!120 = !DILocation(line: 153, column: 77, scope: !100)
!121 = distinct !DISubprogram(name: "Test_Kern_StructNoCopyTrivialMove", linkageName: "_Z33Test_Kern_StructNoCopyTrivialMove23StructNoCopyTrivialMove", scope: !4, file: !4, line: 159, type: !101, scopeLine: 159, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !122)
!122 = !{!123}
!123 = !DILocalVariable(arg: 1, scope: !121, file: !4, line: 159, type: !103)
!124 = distinct !DILifetime(object: !123, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNoCopyTrivialMove)))
!125 = !DILocation(line: 159, column: 74, scope: !121)
!126 = !DILocation(line: 159, column: 77, scope: !121)
!127 = distinct !DISubprogram(name: "Test_Func_StructTrivialCopyNoMove", linkageName: "_Z33Test_Func_StructTrivialCopyNoMove23StructTrivialCopyNoMove", scope: !4, file: !4, line: 165, type: !128, scopeLine: 165, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !143)
!128 = !DISubroutineType(types: !129)
!129 = !{null, !130}
!130 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructTrivialCopyNoMove", file: !4, line: 54, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !131, identifier: "_ZTS23StructTrivialCopyNoMove")
!131 = !{!132, !133, !139}
!132 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !130, file: !4, line: 55, baseType: !5, size: 8)
!133 = !DISubprogram(name: "StructTrivialCopyNoMove", scope: !130, file: !4, line: 56, type: !134, scopeLine: 56, flags: DIFlagPrototyped, spFlags: 0)
!134 = !DISubroutineType(types: !135)
!135 = !{null, !136, !137}
!136 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !130, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!137 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !138, size: 64)
!138 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !130)
!139 = !DISubprogram(name: "StructTrivialCopyNoMove", scope: !130, file: !4, line: 57, type: !140, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDeleted)
!140 = !DISubroutineType(types: !141)
!141 = !{null, !136, !142}
!142 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !130, size: 64)
!143 = !{!144}
!144 = !DILocalVariable(arg: 1, scope: !127, file: !4, line: 165, type: !130)
!145 = distinct !DILifetime(object: !144, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructTrivialCopyNoMove)))
!146 = !DILocation(line: 165, column: 74, scope: !127)
!147 = !DILocation(line: 165, column: 77, scope: !127)
!148 = distinct !DISubprogram(name: "Test_Kern_StructTrivialCopyNoMove", linkageName: "_Z33Test_Kern_StructTrivialCopyNoMove23StructTrivialCopyNoMove", scope: !4, file: !4, line: 171, type: !128, scopeLine: 171, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !149)
!149 = !{!150}
!150 = !DILocalVariable(arg: 1, scope: !148, file: !4, line: 171, type: !130)
!151 = distinct !DILifetime(object: !150, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructTrivialCopyNoMove)))
!152 = !DILocation(line: 171, column: 74, scope: !148)
!153 = !DILocation(line: 171, column: 77, scope: !148)
!154 = distinct !DISubprogram(name: "Test_Func_StructNoCopyNoMove", linkageName: "_Z28Test_Func_StructNoCopyNoMove18StructNoCopyNoMove", scope: !4, file: !4, line: 174, type: !155, scopeLine: 174, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !170)
!155 = !DISubroutineType(types: !156)
!156 = !{null, !157}
!157 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNoCopyNoMove", file: !4, line: 60, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !158, identifier: "_ZTS18StructNoCopyNoMove")
!158 = !{!159, !160, !166}
!159 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !157, file: !4, line: 61, baseType: !5, size: 8)
!160 = !DISubprogram(name: "StructNoCopyNoMove", scope: !157, file: !4, line: 62, type: !161, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDeleted)
!161 = !DISubroutineType(types: !162)
!162 = !{null, !163, !164}
!163 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !157, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!164 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !165, size: 64)
!165 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !157)
!166 = !DISubprogram(name: "StructNoCopyNoMove", scope: !157, file: !4, line: 63, type: !167, scopeLine: 63, flags: DIFlagPrototyped, spFlags: DISPFlagDeleted)
!167 = !DISubroutineType(types: !168)
!168 = !{null, !163, !169}
!169 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !157, size: 64)
!170 = !{!171}
!171 = !DILocalVariable(arg: 1, scope: !154, file: !4, line: 174, type: !157)
!172 = distinct !DILifetime(object: !171, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNoCopyNoMove)))
!173 = !DILocation(line: 174, column: 64, scope: !154)
!174 = !DILocation(line: 174, column: 67, scope: !154)
!175 = distinct !DISubprogram(name: "Test_Kern_StructNoCopyNoMove", linkageName: "_Z28Test_Kern_StructNoCopyNoMove18StructNoCopyNoMove", scope: !4, file: !4, line: 180, type: !155, scopeLine: 180, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !176)
!176 = !{!177}
!177 = !DILocalVariable(arg: 1, scope: !175, file: !4, line: 180, type: !157)
!178 = distinct !DILifetime(object: !177, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNoCopyNoMove)))
!179 = !DILocation(line: 180, column: 64, scope: !175)
!180 = !DILocation(line: 180, column: 67, scope: !175)
!181 = distinct !DISubprogram(name: "Test_Func_Struct2Bytes", linkageName: "_Z22Test_Func_Struct2Bytes12StructNBytesILj2EE", scope: !4, file: !4, line: 186, type: !182, scopeLine: 186, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !193)
!182 = !DISubroutineType(types: !183)
!183 = !{null, !184}
!184 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<2U>", file: !4, line: 75, size: 16, flags: DIFlagTypePassByValue, elements: !185, templateParams: !191, identifier: "_ZTS12StructNBytesILj2EE")
!185 = !{!186, !187}
!186 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !184, file: !4, line: 77, baseType: !5, size: 8)
!187 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !184, file: !4, line: 78, baseType: !188, size: 8, offset: 8)
!188 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 8, elements: !189)
!189 = !{!190}
!190 = !DISubrange(count: 1)
!191 = !{!192}
!192 = !DITemplateValueParameter(name: "N", type: !16, value: i32 2)
!193 = !{!194}
!194 = !DILocalVariable(arg: 1, scope: !181, file: !4, line: 186, type: !184)
!195 = distinct !DILifetime(object: !194, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes)))
!196 = !DILocation(line: 186, column: 55, scope: !181)
!197 = !DILocation(line: 186, column: 58, scope: !181)
!198 = distinct !DISubprogram(name: "Test_Kern_Struct2Bytes", linkageName: "_Z22Test_Kern_Struct2Bytes12StructNBytesILj2EE", scope: !4, file: !4, line: 192, type: !182, scopeLine: 192, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !199)
!199 = !{!200}
!200 = !DILocalVariable(arg: 1, scope: !198, file: !4, line: 192, type: !184)
!201 = distinct !DILifetime(object: !200, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes)))
!202 = !DILocation(line: 192, column: 55, scope: !198)
!203 = !DILocation(line: 192, column: 58, scope: !198)
!204 = distinct !DISubprogram(name: "Test_Func_Struct3Bytes", linkageName: "_Z22Test_Func_Struct3Bytes12StructNBytesILj3EE", scope: !4, file: !4, line: 201, type: !205, scopeLine: 201, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !216)
!205 = !DISubroutineType(types: !206)
!206 = !{null, !207}
!207 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<3U>", file: !4, line: 75, size: 24, flags: DIFlagTypePassByValue, elements: !208, templateParams: !214, identifier: "_ZTS12StructNBytesILj3EE")
!208 = !{!209, !210}
!209 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !207, file: !4, line: 77, baseType: !5, size: 8)
!210 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !207, file: !4, line: 78, baseType: !211, size: 16, offset: 8)
!211 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 16, elements: !212)
!212 = !{!213}
!213 = !DISubrange(count: 2)
!214 = !{!215}
!215 = !DITemplateValueParameter(name: "N", type: !16, value: i32 3)
!216 = !{!217}
!217 = !DILocalVariable(arg: 1, scope: !204, file: !4, line: 201, type: !207)
!218 = distinct !DILifetime(object: !217, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.0)))
!219 = !DILocation(line: 201, column: 55, scope: !204)
!220 = !DILocation(line: 201, column: 58, scope: !204)
!221 = distinct !DISubprogram(name: "Test_Kern_Struct3Bytes", linkageName: "_Z22Test_Kern_Struct3Bytes12StructNBytesILj3EE", scope: !4, file: !4, line: 207, type: !205, scopeLine: 207, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !222)
!222 = !{!223}
!223 = !DILocalVariable(arg: 1, scope: !221, file: !4, line: 207, type: !207)
!224 = distinct !DILifetime(object: !223, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.0)))
!225 = !DILocation(line: 207, column: 55, scope: !221)
!226 = !DILocation(line: 207, column: 58, scope: !221)
!227 = distinct !DISubprogram(name: "Test_Func_Struct4Bytes", linkageName: "_Z22Test_Func_Struct4Bytes12StructNBytesILj4EE", scope: !4, file: !4, line: 213, type: !228, scopeLine: 213, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !239)
!228 = !DISubroutineType(types: !229)
!229 = !{null, !230}
!230 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<4U>", file: !4, line: 75, size: 32, flags: DIFlagTypePassByValue, elements: !231, templateParams: !237, identifier: "_ZTS12StructNBytesILj4EE")
!231 = !{!232, !233}
!232 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !230, file: !4, line: 77, baseType: !5, size: 8)
!233 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !230, file: !4, line: 78, baseType: !234, size: 24, offset: 8)
!234 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 24, elements: !235)
!235 = !{!236}
!236 = !DISubrange(count: 3)
!237 = !{!238}
!238 = !DITemplateValueParameter(name: "N", type: !16, value: i32 4)
!239 = !{!240}
!240 = !DILocalVariable(arg: 1, scope: !227, file: !4, line: 213, type: !230)
!241 = distinct !DILifetime(object: !240, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.1)))
!242 = !DILocation(line: 213, column: 55, scope: !227)
!243 = !DILocation(line: 213, column: 58, scope: !227)
!244 = distinct !DISubprogram(name: "Test_Kern_Struct4Bytes", linkageName: "_Z22Test_Kern_Struct4Bytes12StructNBytesILj4EE", scope: !4, file: !4, line: 218, type: !228, scopeLine: 218, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !245)
!245 = !{!246}
!246 = !DILocalVariable(arg: 1, scope: !244, file: !4, line: 218, type: !230)
!247 = distinct !DILifetime(object: !246, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.1)))
!248 = !DILocation(line: 218, column: 55, scope: !244)
!249 = !DILocation(line: 218, column: 58, scope: !244)
!250 = distinct !DISubprogram(name: "Test_Func_Struct5Bytes", linkageName: "_Z22Test_Func_Struct5Bytes12StructNBytesILj5EE", scope: !4, file: !4, line: 224, type: !251, scopeLine: 224, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !262)
!251 = !DISubroutineType(types: !252)
!252 = !{null, !253}
!253 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<5U>", file: !4, line: 75, size: 40, flags: DIFlagTypePassByValue, elements: !254, templateParams: !260, identifier: "_ZTS12StructNBytesILj5EE")
!254 = !{!255, !256}
!255 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !253, file: !4, line: 77, baseType: !5, size: 8)
!256 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !253, file: !4, line: 78, baseType: !257, size: 32, offset: 8)
!257 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 32, elements: !258)
!258 = !{!259}
!259 = !DISubrange(count: 4)
!260 = !{!261}
!261 = !DITemplateValueParameter(name: "N", type: !16, value: i32 5)
!262 = !{!263}
!263 = !DILocalVariable(arg: 1, scope: !250, file: !4, line: 224, type: !253)
!264 = distinct !DILifetime(object: !263, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.2)))
!265 = !DILocation(line: 224, column: 55, scope: !250)
!266 = !DILocation(line: 224, column: 58, scope: !250)
!267 = distinct !DISubprogram(name: "Test_Kern_Struct5Bytes", linkageName: "_Z22Test_Kern_Struct5Bytes12StructNBytesILj5EE", scope: !4, file: !4, line: 230, type: !251, scopeLine: 230, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !268)
!268 = !{!269}
!269 = !DILocalVariable(arg: 1, scope: !267, file: !4, line: 230, type: !253)
!270 = distinct !DILifetime(object: !269, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.2)))
!271 = !DILocation(line: 230, column: 55, scope: !267)
!272 = !DILocation(line: 230, column: 58, scope: !267)
!273 = distinct !DISubprogram(name: "Test_Func_Struct6Bytes", linkageName: "_Z22Test_Func_Struct6Bytes12StructNBytesILj6EE", scope: !4, file: !4, line: 236, type: !274, scopeLine: 236, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !285)
!274 = !DISubroutineType(types: !275)
!275 = !{null, !276}
!276 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<6U>", file: !4, line: 75, size: 48, flags: DIFlagTypePassByValue, elements: !277, templateParams: !283, identifier: "_ZTS12StructNBytesILj6EE")
!277 = !{!278, !279}
!278 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !276, file: !4, line: 77, baseType: !5, size: 8)
!279 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !276, file: !4, line: 78, baseType: !280, size: 40, offset: 8)
!280 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 40, elements: !281)
!281 = !{!282}
!282 = !DISubrange(count: 5)
!283 = !{!284}
!284 = !DITemplateValueParameter(name: "N", type: !16, value: i32 6)
!285 = !{!286}
!286 = !DILocalVariable(arg: 1, scope: !273, file: !4, line: 236, type: !276)
!287 = distinct !DILifetime(object: !286, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.3)))
!288 = !DILocation(line: 236, column: 55, scope: !273)
!289 = !DILocation(line: 236, column: 58, scope: !273)
!290 = distinct !DISubprogram(name: "Test_Kern_Struct6Bytes", linkageName: "_Z22Test_Kern_Struct6Bytes12StructNBytesILj6EE", scope: !4, file: !4, line: 242, type: !274, scopeLine: 242, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !291)
!291 = !{!292}
!292 = !DILocalVariable(arg: 1, scope: !290, file: !4, line: 242, type: !276)
!293 = distinct !DILifetime(object: !292, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.3)))
!294 = !DILocation(line: 242, column: 55, scope: !290)
!295 = !DILocation(line: 242, column: 58, scope: !290)
!296 = distinct !DISubprogram(name: "Test_Func_Struct7Bytes", linkageName: "_Z22Test_Func_Struct7Bytes12StructNBytesILj7EE", scope: !4, file: !4, line: 248, type: !297, scopeLine: 248, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !308)
!297 = !DISubroutineType(types: !298)
!298 = !{null, !299}
!299 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<7U>", file: !4, line: 75, size: 56, flags: DIFlagTypePassByValue, elements: !300, templateParams: !306, identifier: "_ZTS12StructNBytesILj7EE")
!300 = !{!301, !302}
!301 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !299, file: !4, line: 77, baseType: !5, size: 8)
!302 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !299, file: !4, line: 78, baseType: !303, size: 48, offset: 8)
!303 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 48, elements: !304)
!304 = !{!305}
!305 = !DISubrange(count: 6)
!306 = !{!307}
!307 = !DITemplateValueParameter(name: "N", type: !16, value: i32 7)
!308 = !{!309}
!309 = !DILocalVariable(arg: 1, scope: !296, file: !4, line: 248, type: !299)
!310 = distinct !DILifetime(object: !309, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.4)))
!311 = !DILocation(line: 248, column: 55, scope: !296)
!312 = !DILocation(line: 248, column: 58, scope: !296)
!313 = distinct !DISubprogram(name: "Test_Kern_Struct7Bytes", linkageName: "_Z22Test_Kern_Struct7Bytes12StructNBytesILj7EE", scope: !4, file: !4, line: 254, type: !297, scopeLine: 254, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !314)
!314 = !{!315}
!315 = !DILocalVariable(arg: 1, scope: !313, file: !4, line: 254, type: !299)
!316 = distinct !DILifetime(object: !315, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.4)))
!317 = !DILocation(line: 254, column: 55, scope: !313)
!318 = !DILocation(line: 254, column: 58, scope: !313)
!319 = distinct !DISubprogram(name: "Test_Func_Struct8Bytes", linkageName: "_Z22Test_Func_Struct8Bytes12StructNBytesILj8EE", scope: !4, file: !4, line: 260, type: !320, scopeLine: 260, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !331)
!320 = !DISubroutineType(types: !321)
!321 = !{null, !322}
!322 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<8U>", file: !4, line: 75, size: 64, flags: DIFlagTypePassByValue, elements: !323, templateParams: !329, identifier: "_ZTS12StructNBytesILj8EE")
!323 = !{!324, !325}
!324 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !322, file: !4, line: 77, baseType: !5, size: 8)
!325 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !322, file: !4, line: 78, baseType: !326, size: 56, offset: 8)
!326 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 56, elements: !327)
!327 = !{!328}
!328 = !DISubrange(count: 7)
!329 = !{!330}
!330 = !DITemplateValueParameter(name: "N", type: !16, value: i32 8)
!331 = !{!332}
!332 = !DILocalVariable(arg: 1, scope: !319, file: !4, line: 260, type: !322)
!333 = distinct !DILifetime(object: !332, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.5)))
!334 = !DILocation(line: 260, column: 55, scope: !319)
!335 = !DILocation(line: 260, column: 58, scope: !319)
!336 = distinct !DISubprogram(name: "Test_Kern_Struct8Bytes", linkageName: "_Z22Test_Kern_Struct8Bytes12StructNBytesILj8EE", scope: !4, file: !4, line: 266, type: !320, scopeLine: 266, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !337)
!337 = !{!338}
!338 = !DILocalVariable(arg: 1, scope: !336, file: !4, line: 266, type: !322)
!339 = distinct !DILifetime(object: !338, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.5)))
!340 = !DILocation(line: 266, column: 55, scope: !336)
!341 = !DILocation(line: 266, column: 58, scope: !336)
!342 = distinct !DISubprogram(name: "Test_Func_Struct9Bytes", linkageName: "_Z22Test_Func_Struct9Bytes12StructNBytesILj9EE", scope: !4, file: !4, line: 273, type: !343, scopeLine: 273, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !354)
!343 = !DISubroutineType(types: !344)
!344 = !{null, !345}
!345 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<9U>", file: !4, line: 75, size: 72, flags: DIFlagTypePassByValue, elements: !346, templateParams: !352, identifier: "_ZTS12StructNBytesILj9EE")
!346 = !{!347, !348}
!347 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !345, file: !4, line: 77, baseType: !5, size: 8)
!348 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !345, file: !4, line: 78, baseType: !349, size: 64, offset: 8)
!349 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 64, elements: !350)
!350 = !{!351}
!351 = !DISubrange(count: 8)
!352 = !{!353}
!353 = !DITemplateValueParameter(name: "N", type: !16, value: i32 9)
!354 = !{!355}
!355 = !DILocalVariable(arg: 1, scope: !342, file: !4, line: 273, type: !345)
!356 = distinct !DILifetime(object: !355, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.6)))
!357 = !DILocation(line: 273, column: 55, scope: !342)
!358 = !DILocation(line: 273, column: 58, scope: !342)
!359 = distinct !DISubprogram(name: "Test_Kern_Struct9Bytes", linkageName: "_Z22Test_Kern_Struct9Bytes12StructNBytesILj9EE", scope: !4, file: !4, line: 279, type: !343, scopeLine: 279, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !360)
!360 = !{!361}
!361 = !DILocalVariable(arg: 1, scope: !359, file: !4, line: 279, type: !345)
!362 = distinct !DILifetime(object: !361, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.6)))
!363 = !DILocation(line: 279, column: 55, scope: !359)
!364 = !DILocation(line: 279, column: 58, scope: !359)
!365 = distinct !DISubprogram(name: "Test_Func_Struct64Bytes", linkageName: "_Z23Test_Func_Struct64Bytes12StructNBytesILj64EE", scope: !4, file: !4, line: 283, type: !366, scopeLine: 283, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !377)
!366 = !DISubroutineType(types: !367)
!367 = !{null, !368}
!368 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructNBytes<64U>", file: !4, line: 75, size: 512, flags: DIFlagTypePassByValue, elements: !369, templateParams: !375, identifier: "_ZTS12StructNBytesILj64EE")
!369 = !{!370, !371}
!370 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !368, file: !4, line: 77, baseType: !5, size: 8)
!371 = !DIDerivedType(tag: DW_TAG_member, name: "Elements", scope: !368, file: !4, line: 78, baseType: !372, size: 504, offset: 8)
!372 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 504, elements: !373)
!373 = !{!374}
!374 = !DISubrange(count: 63)
!375 = !{!376}
!376 = !DITemplateValueParameter(name: "N", type: !16, value: i32 64)
!377 = !{!378}
!378 = !DILocalVariable(arg: 1, scope: !365, file: !4, line: 283, type: !368)
!379 = distinct !DILifetime(object: !378, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.7)))
!380 = !DILocation(line: 283, column: 57, scope: !365)
!381 = !DILocation(line: 283, column: 60, scope: !365)
!382 = distinct !DISubprogram(name: "Test_Kern_Struct64Bytes", linkageName: "_Z23Test_Kern_Struct64Bytes12StructNBytesILj64EE", scope: !4, file: !4, line: 289, type: !366, scopeLine: 289, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !383)
!383 = !{!384}
!384 = !DILocalVariable(arg: 1, scope: !382, file: !4, line: 289, type: !368)
!385 = distinct !DILifetime(object: !384, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructNBytes.7)))
!386 = !DILocation(line: 289, column: 57, scope: !382)
!387 = !DILocation(line: 289, column: 60, scope: !382)
!388 = distinct !DISubprogram(name: "Test_Func_Int8T", linkageName: "_Z15Test_Func_Int8Tc", scope: !4, file: !4, line: 294, type: !389, scopeLine: 294, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !391)
!389 = !DISubroutineType(types: !390)
!390 = !{null, !5}
!391 = !{!392}
!392 = !DILocalVariable(arg: 1, scope: !388, file: !4, line: 294, type: !5)
!393 = distinct !DILifetime(object: !392, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!394 = !DILocation(line: 294, column: 39, scope: !388)
!395 = !DILocation(line: 294, column: 42, scope: !388)
!396 = distinct !DISubprogram(name: "Test_Kern_Int8T", linkageName: "_Z15Test_Kern_Int8Tc", scope: !4, file: !4, line: 299, type: !389, scopeLine: 299, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !397)
!397 = !{!398}
!398 = !DILocalVariable(arg: 1, scope: !396, file: !4, line: 299, type: !5)
!399 = distinct !DILifetime(object: !398, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!400 = !DILocation(line: 299, column: 39, scope: !396)
!401 = !DILocation(line: 299, column: 42, scope: !396)
!402 = distinct !DISubprogram(name: "Test_Func_UInt8T", linkageName: "_Z16Test_Func_UInt8Th", scope: !4, file: !4, line: 304, type: !403, scopeLine: 304, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !405)
!403 = !DISubroutineType(types: !404)
!404 = !{null, !8}
!405 = !{!406}
!406 = !DILocalVariable(arg: 1, scope: !402, file: !4, line: 304, type: !8)
!407 = distinct !DILifetime(object: !406, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!408 = !DILocation(line: 304, column: 41, scope: !402)
!409 = !DILocation(line: 304, column: 44, scope: !402)
!410 = distinct !DISubprogram(name: "Test_Kern_UInt8T", linkageName: "_Z16Test_Kern_UInt8Th", scope: !4, file: !4, line: 309, type: !403, scopeLine: 309, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !411)
!411 = !{!412}
!412 = !DILocalVariable(arg: 1, scope: !410, file: !4, line: 309, type: !8)
!413 = distinct !DILifetime(object: !412, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!414 = !DILocation(line: 309, column: 41, scope: !410)
!415 = !DILocation(line: 309, column: 44, scope: !410)
!416 = distinct !DISubprogram(name: "Test_Func_Int16T", linkageName: "_Z16Test_Func_Int16Ts", scope: !4, file: !4, line: 314, type: !417, scopeLine: 314, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !419)
!417 = !DISubroutineType(types: !418)
!418 = !{null, !10}
!419 = !{!420}
!420 = !DILocalVariable(arg: 1, scope: !416, file: !4, line: 314, type: !10)
!421 = distinct !DILifetime(object: !420, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!422 = !DILocation(line: 314, column: 41, scope: !416)
!423 = !DILocation(line: 314, column: 44, scope: !416)
!424 = distinct !DISubprogram(name: "Test_Kern_Int16T", linkageName: "_Z16Test_Kern_Int16Ts", scope: !4, file: !4, line: 319, type: !417, scopeLine: 319, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !425)
!425 = !{!426}
!426 = !DILocalVariable(arg: 1, scope: !424, file: !4, line: 319, type: !10)
!427 = distinct !DILifetime(object: !426, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!428 = !DILocation(line: 319, column: 41, scope: !424)
!429 = !DILocation(line: 319, column: 44, scope: !424)
!430 = distinct !DISubprogram(name: "Test_Func_UInt16T", linkageName: "_Z17Test_Func_UInt16Tt", scope: !4, file: !4, line: 324, type: !431, scopeLine: 324, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !433)
!431 = !DISubroutineType(types: !432)
!432 = !{null, !12}
!433 = !{!434}
!434 = !DILocalVariable(arg: 1, scope: !430, file: !4, line: 324, type: !12)
!435 = distinct !DILifetime(object: !434, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!436 = !DILocation(line: 324, column: 43, scope: !430)
!437 = !DILocation(line: 324, column: 46, scope: !430)
!438 = distinct !DISubprogram(name: "Test_Kern_UInt16T", linkageName: "_Z17Test_Kern_UInt16Tt", scope: !4, file: !4, line: 329, type: !431, scopeLine: 329, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !439)
!439 = !{!440}
!440 = !DILocalVariable(arg: 1, scope: !438, file: !4, line: 329, type: !12)
!441 = distinct !DILifetime(object: !440, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!442 = !DILocation(line: 329, column: 43, scope: !438)
!443 = !DILocation(line: 329, column: 46, scope: !438)
!444 = distinct !DISubprogram(name: "Test_Func_Int32T", linkageName: "_Z16Test_Func_Int32Ti", scope: !4, file: !4, line: 334, type: !445, scopeLine: 334, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !447)
!445 = !DISubroutineType(types: !446)
!446 = !{null, !14}
!447 = !{!448}
!448 = !DILocalVariable(arg: 1, scope: !444, file: !4, line: 334, type: !14)
!449 = distinct !DILifetime(object: !448, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!450 = !DILocation(line: 334, column: 41, scope: !444)
!451 = !DILocation(line: 334, column: 44, scope: !444)
!452 = distinct !DISubprogram(name: "Test_Kern_Int32T", linkageName: "_Z16Test_Kern_Int32Ti", scope: !4, file: !4, line: 339, type: !445, scopeLine: 339, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !453)
!453 = !{!454}
!454 = !DILocalVariable(arg: 1, scope: !452, file: !4, line: 339, type: !14)
!455 = distinct !DILifetime(object: !454, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!456 = !DILocation(line: 339, column: 41, scope: !452)
!457 = !DILocation(line: 339, column: 44, scope: !452)
!458 = distinct !DISubprogram(name: "Test_Func_UInt32T", linkageName: "_Z17Test_Func_UInt32Tj", scope: !4, file: !4, line: 344, type: !459, scopeLine: 344, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !461)
!459 = !DISubroutineType(types: !460)
!460 = !{null, !16}
!461 = !{!462}
!462 = !DILocalVariable(arg: 1, scope: !458, file: !4, line: 344, type: !16)
!463 = distinct !DILifetime(object: !462, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!464 = !DILocation(line: 344, column: 43, scope: !458)
!465 = !DILocation(line: 344, column: 46, scope: !458)
!466 = distinct !DISubprogram(name: "Test_Kern_UInt32T", linkageName: "_Z17Test_Kern_UInt32Tj", scope: !4, file: !4, line: 349, type: !459, scopeLine: 349, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !467)
!467 = !{!468}
!468 = !DILocalVariable(arg: 1, scope: !466, file: !4, line: 349, type: !16)
!469 = distinct !DILifetime(object: !468, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!470 = !DILocation(line: 349, column: 43, scope: !466)
!471 = !DILocation(line: 349, column: 46, scope: !466)
!472 = distinct !DISubprogram(name: "Test_Func_Int64T", linkageName: "_Z16Test_Func_Int64Tl", scope: !4, file: !4, line: 354, type: !473, scopeLine: 354, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !475)
!473 = !DISubroutineType(types: !474)
!474 = !{null, !18}
!475 = !{!476}
!476 = !DILocalVariable(arg: 1, scope: !472, file: !4, line: 354, type: !18)
!477 = distinct !DILifetime(object: !476, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!478 = !DILocation(line: 354, column: 41, scope: !472)
!479 = !DILocation(line: 354, column: 44, scope: !472)
!480 = distinct !DISubprogram(name: "Test_Kern_Int64T", linkageName: "_Z16Test_Kern_Int64Tl", scope: !4, file: !4, line: 359, type: !473, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !481)
!481 = !{!482}
!482 = !DILocalVariable(arg: 1, scope: !480, file: !4, line: 359, type: !18)
!483 = distinct !DILifetime(object: !482, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!484 = !DILocation(line: 359, column: 41, scope: !480)
!485 = !DILocation(line: 359, column: 44, scope: !480)
!486 = distinct !DISubprogram(name: "Test_Func_UInt64T", linkageName: "_Z17Test_Func_UInt64Tm", scope: !4, file: !4, line: 364, type: !487, scopeLine: 364, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !489)
!487 = !DISubroutineType(types: !488)
!488 = !{null, !20}
!489 = !{!490}
!490 = !DILocalVariable(arg: 1, scope: !486, file: !4, line: 364, type: !20)
!491 = distinct !DILifetime(object: !490, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!492 = !DILocation(line: 364, column: 43, scope: !486)
!493 = !DILocation(line: 364, column: 46, scope: !486)
!494 = distinct !DISubprogram(name: "Test_Kern_UInt64T", linkageName: "_Z17Test_Kern_UInt64Tm", scope: !4, file: !4, line: 369, type: !487, scopeLine: 369, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !495)
!495 = !{!496}
!496 = !DILocalVariable(arg: 1, scope: !494, file: !4, line: 369, type: !20)
!497 = distinct !DILifetime(object: !496, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!498 = !DILocation(line: 369, column: 43, scope: !494)
!499 = !DILocation(line: 369, column: 46, scope: !494)
!500 = distinct !DISubprogram(name: "Test_Func_EnumInt8T", linkageName: "_Z19Test_Func_EnumInt8T9EnumInt8T", scope: !4, file: !4, line: 374, type: !501, scopeLine: 374, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !503)
!501 = !DISubroutineType(types: !502)
!502 = !{null, !3}
!503 = !{!504}
!504 = !DILocalVariable(arg: 1, scope: !500, file: !4, line: 374, type: !3)
!505 = distinct !DILifetime(object: !504, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!506 = !DILocation(line: 374, column: 46, scope: !500)
!507 = !DILocation(line: 374, column: 49, scope: !500)
!508 = distinct !DISubprogram(name: "Test_Kern_EnumInt8T", linkageName: "_Z19Test_Kern_EnumInt8T9EnumInt8T", scope: !4, file: !4, line: 379, type: !501, scopeLine: 379, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !509)
!509 = !{!510}
!510 = !DILocalVariable(arg: 1, scope: !508, file: !4, line: 379, type: !3)
!511 = distinct !DILifetime(object: !510, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!512 = !DILocation(line: 379, column: 46, scope: !508)
!513 = !DILocation(line: 379, column: 49, scope: !508)
!514 = distinct !DISubprogram(name: "Test_Func_EnumUInt8T", linkageName: "_Z20Test_Func_EnumUInt8T10EnumUInt8T", scope: !4, file: !4, line: 384, type: !515, scopeLine: 384, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !517)
!515 = !DISubroutineType(types: !516)
!516 = !{null, !7}
!517 = !{!518}
!518 = !DILocalVariable(arg: 1, scope: !514, file: !4, line: 384, type: !7)
!519 = distinct !DILifetime(object: !518, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!520 = !DILocation(line: 384, column: 48, scope: !514)
!521 = !DILocation(line: 384, column: 51, scope: !514)
!522 = distinct !DISubprogram(name: "Test_Kern_EnumUInt8T", linkageName: "_Z20Test_Kern_EnumUInt8T10EnumUInt8T", scope: !4, file: !4, line: 389, type: !515, scopeLine: 389, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !523)
!523 = !{!524}
!524 = !DILocalVariable(arg: 1, scope: !522, file: !4, line: 389, type: !7)
!525 = distinct !DILifetime(object: !524, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!526 = !DILocation(line: 389, column: 48, scope: !522)
!527 = !DILocation(line: 389, column: 51, scope: !522)
!528 = distinct !DISubprogram(name: "Test_Func_EnumInt16T", linkageName: "_Z20Test_Func_EnumInt16T10EnumInt16T", scope: !4, file: !4, line: 394, type: !529, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !531)
!529 = !DISubroutineType(types: !530)
!530 = !{null, !9}
!531 = !{!532}
!532 = !DILocalVariable(arg: 1, scope: !528, file: !4, line: 394, type: !9)
!533 = distinct !DILifetime(object: !532, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!534 = !DILocation(line: 394, column: 48, scope: !528)
!535 = !DILocation(line: 394, column: 51, scope: !528)
!536 = distinct !DISubprogram(name: "Test_Kern_EnumInt16T", linkageName: "_Z20Test_Kern_EnumInt16T10EnumInt16T", scope: !4, file: !4, line: 399, type: !529, scopeLine: 399, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !537)
!537 = !{!538}
!538 = !DILocalVariable(arg: 1, scope: !536, file: !4, line: 399, type: !9)
!539 = distinct !DILifetime(object: !538, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!540 = !DILocation(line: 399, column: 48, scope: !536)
!541 = !DILocation(line: 399, column: 51, scope: !536)
!542 = distinct !DISubprogram(name: "Test_Func_EnumUInt16T", linkageName: "_Z21Test_Func_EnumUInt16T11EnumUInt16T", scope: !4, file: !4, line: 404, type: !543, scopeLine: 404, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !545)
!543 = !DISubroutineType(types: !544)
!544 = !{null, !11}
!545 = !{!546}
!546 = !DILocalVariable(arg: 1, scope: !542, file: !4, line: 404, type: !11)
!547 = distinct !DILifetime(object: !546, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!548 = !DILocation(line: 404, column: 50, scope: !542)
!549 = !DILocation(line: 404, column: 53, scope: !542)
!550 = distinct !DISubprogram(name: "Test_Kern_EnumUInt16T", linkageName: "_Z21Test_Kern_EnumUInt16T11EnumUInt16T", scope: !4, file: !4, line: 409, type: !543, scopeLine: 409, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !551)
!551 = !{!552}
!552 = !DILocalVariable(arg: 1, scope: !550, file: !4, line: 409, type: !11)
!553 = distinct !DILifetime(object: !552, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i16)))
!554 = !DILocation(line: 409, column: 50, scope: !550)
!555 = !DILocation(line: 409, column: 53, scope: !550)
!556 = distinct !DISubprogram(name: "Test_Func_EnumInt32T", linkageName: "_Z20Test_Func_EnumInt32T10EnumInt32T", scope: !4, file: !4, line: 414, type: !557, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !559)
!557 = !DISubroutineType(types: !558)
!558 = !{null, !13}
!559 = !{!560}
!560 = !DILocalVariable(arg: 1, scope: !556, file: !4, line: 414, type: !13)
!561 = distinct !DILifetime(object: !560, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!562 = !DILocation(line: 414, column: 48, scope: !556)
!563 = !DILocation(line: 414, column: 51, scope: !556)
!564 = distinct !DISubprogram(name: "Test_Kern_EnumInt32T", linkageName: "_Z20Test_Kern_EnumInt32T10EnumInt32T", scope: !4, file: !4, line: 419, type: !557, scopeLine: 419, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !565)
!565 = !{!566}
!566 = !DILocalVariable(arg: 1, scope: !564, file: !4, line: 419, type: !13)
!567 = distinct !DILifetime(object: !566, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!568 = !DILocation(line: 419, column: 48, scope: !564)
!569 = !DILocation(line: 419, column: 51, scope: !564)
!570 = distinct !DISubprogram(name: "Test_Func_EnumUInt32T", linkageName: "_Z21Test_Func_EnumUInt32T11EnumUInt32T", scope: !4, file: !4, line: 424, type: !571, scopeLine: 424, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !573)
!571 = !DISubroutineType(types: !572)
!572 = !{null, !15}
!573 = !{!574}
!574 = !DILocalVariable(arg: 1, scope: !570, file: !4, line: 424, type: !15)
!575 = distinct !DILifetime(object: !574, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!576 = !DILocation(line: 424, column: 50, scope: !570)
!577 = !DILocation(line: 424, column: 53, scope: !570)
!578 = distinct !DISubprogram(name: "Test_Kern_EnumUInt32T", linkageName: "_Z21Test_Kern_EnumUInt32T11EnumUInt32T", scope: !4, file: !4, line: 429, type: !571, scopeLine: 429, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !579)
!579 = !{!580}
!580 = !DILocalVariable(arg: 1, scope: !578, file: !4, line: 429, type: !15)
!581 = distinct !DILifetime(object: !580, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!582 = !DILocation(line: 429, column: 50, scope: !578)
!583 = !DILocation(line: 429, column: 53, scope: !578)
!584 = distinct !DISubprogram(name: "Test_Func_EnumInt64T", linkageName: "_Z20Test_Func_EnumInt64T10EnumInt64T", scope: !4, file: !4, line: 434, type: !585, scopeLine: 434, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !587)
!585 = !DISubroutineType(types: !586)
!586 = !{null, !17}
!587 = !{!588}
!588 = !DILocalVariable(arg: 1, scope: !584, file: !4, line: 434, type: !17)
!589 = distinct !DILifetime(object: !588, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!590 = !DILocation(line: 434, column: 48, scope: !584)
!591 = !DILocation(line: 434, column: 51, scope: !584)
!592 = distinct !DISubprogram(name: "Test_Kern_EnumInt64T", linkageName: "_Z20Test_Kern_EnumInt64T10EnumInt64T", scope: !4, file: !4, line: 439, type: !585, scopeLine: 439, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !593)
!593 = !{!594}
!594 = !DILocalVariable(arg: 1, scope: !592, file: !4, line: 439, type: !17)
!595 = distinct !DILifetime(object: !594, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!596 = !DILocation(line: 439, column: 48, scope: !592)
!597 = !DILocation(line: 439, column: 51, scope: !592)
!598 = distinct !DISubprogram(name: "Test_Func_EnumUInt64T", linkageName: "_Z21Test_Func_EnumUInt64T11EnumUInt64T", scope: !4, file: !4, line: 444, type: !599, scopeLine: 444, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !601)
!599 = !DISubroutineType(types: !600)
!600 = !{null, !19}
!601 = !{!602}
!602 = !DILocalVariable(arg: 1, scope: !598, file: !4, line: 444, type: !19)
!603 = distinct !DILifetime(object: !602, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!604 = !DILocation(line: 444, column: 50, scope: !598)
!605 = !DILocation(line: 444, column: 53, scope: !598)
!606 = distinct !DISubprogram(name: "Test_Kern_EnumUInt64T", linkageName: "_Z21Test_Kern_EnumUInt64T11EnumUInt64T", scope: !4, file: !4, line: 449, type: !599, scopeLine: 449, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !607)
!607 = !{!608}
!608 = !DILocalVariable(arg: 1, scope: !606, file: !4, line: 449, type: !19)
!609 = distinct !DILifetime(object: !608, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!610 = !DILocation(line: 449, column: 50, scope: !606)
!611 = !DILocation(line: 449, column: 53, scope: !606)
!612 = distinct !DISubprogram(name: "Test_Func_PromotableInteger", linkageName: "_Z27Test_Func_PromotableIntegerb", scope: !4, file: !4, line: 455, type: !613, scopeLine: 455, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !616)
!613 = !DISubroutineType(types: !614)
!614 = !{null, !615}
!615 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!616 = !{!617}
!617 = !DILocalVariable(arg: 1, scope: !612, file: !4, line: 455, type: !615)
!618 = distinct !DILifetime(object: !617, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!619 = !DILocation(line: 455, column: 49, scope: !612)
!620 = !DILocation(line: 455, column: 52, scope: !612)
!621 = distinct !DISubprogram(name: "Test_Kern_PromotableInteger", linkageName: "_Z27Test_Kern_PromotableIntegerb", scope: !4, file: !4, line: 461, type: !613, scopeLine: 461, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !622)
!622 = !{!623}
!623 = !DILocalVariable(arg: 1, scope: !621, file: !4, line: 461, type: !615)
!624 = distinct !DILifetime(object: !623, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i8)))
!625 = !DILocation(line: 461, column: 49, scope: !621)
!626 = !DILocation(line: 461, column: 52, scope: !621)
!627 = distinct !DISubprogram(name: "Test_Func_Pointer", linkageName: "_Z17Test_Func_PointerPi", scope: !4, file: !4, line: 466, type: !628, scopeLine: 466, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !631)
!628 = !DISubroutineType(types: !629)
!629 = !{null, !630}
!630 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!631 = !{!632}
!632 = !DILocalVariable(arg: 1, scope: !627, file: !4, line: 466, type: !630)
!633 = distinct !DILifetime(object: !632, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr)))
!634 = !DILocation(line: 466, column: 44, scope: !627)
!635 = !DILocation(line: 466, column: 47, scope: !627)
!636 = distinct !DISubprogram(name: "Test_Kern_Pointer", linkageName: "_Z17Test_Kern_PointerPi", scope: !4, file: !4, line: 473, type: !628, scopeLine: 473, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !637)
!637 = !{!638}
!638 = !DILocalVariable(arg: 1, scope: !636, file: !4, line: 473, type: !630)
!639 = distinct !DILifetime(object: !638, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr)))
!640 = !DILocation(line: 473, column: 44, scope: !636)
!641 = !DILocation(line: 473, column: 47, scope: !636)
!642 = distinct !DISubprogram(name: "Test_Func_Reference", linkageName: "_Z19Test_Func_ReferenceRi", scope: !4, file: !4, line: 478, type: !643, scopeLine: 478, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !646)
!643 = !DISubroutineType(types: !644)
!644 = !{null, !645}
!645 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !14, size: 64)
!646 = !{!647}
!647 = !DILocalVariable(arg: 1, scope: !642, file: !4, line: 478, type: !645)
!648 = distinct !DILifetime(object: !647, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr)))
!649 = !DILocation(line: 478, column: 46, scope: !642)
!650 = !DILocation(line: 478, column: 49, scope: !642)
!651 = distinct !DISubprogram(name: "Test_Kern_Reference", linkageName: "_Z19Test_Kern_ReferenceRi", scope: !4, file: !4, line: 485, type: !643, scopeLine: 485, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !652)
!652 = !{!653}
!653 = !DILocalVariable(arg: 1, scope: !651, file: !4, line: 485, type: !645)
!654 = distinct !DILifetime(object: !653, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr)))
!655 = !DILocation(line: 485, column: 46, scope: !651)
!656 = !DILocation(line: 485, column: 49, scope: !651)
!657 = distinct !DISubprogram(name: "Test_Func_StructSinglePointerElement", linkageName: "_Z36Test_Func_StructSinglePointerElement26StructSinglePointerElement", scope: !4, file: !4, line: 490, type: !658, scopeLine: 490, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !663)
!658 = !DISubroutineType(types: !659)
!659 = !{null, !660}
!660 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructSinglePointerElement", file: !4, line: 89, size: 64, flags: DIFlagTypePassByValue, elements: !661, identifier: "_ZTS26StructSinglePointerElement")
!661 = !{!662}
!662 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !660, file: !4, line: 90, baseType: !630, size: 64)
!663 = !{!664}
!664 = !DILocalVariable(arg: 1, scope: !657, file: !4, line: 490, type: !660)
!665 = distinct !DILifetime(object: !664, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructSinglePointerElement)))
!666 = !DILocation(line: 490, column: 80, scope: !657)
!667 = !DILocation(line: 490, column: 83, scope: !657)
!668 = distinct !DISubprogram(name: "Test_Kern_StructSinglePointerElement", linkageName: "_Z36Test_Kern_StructSinglePointerElement26StructSinglePointerElement", scope: !4, file: !4, line: 495, type: !658, scopeLine: 495, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !669)
!669 = !{!670}
!670 = !DILocalVariable(arg: 1, scope: !668, file: !4, line: 495, type: !660)
!671 = distinct !DILifetime(object: !670, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructSinglePointerElement)))
!672 = !DILocation(line: 495, column: 80, scope: !668)
!673 = !DILocation(line: 495, column: 83, scope: !668)
!674 = distinct !DISubprogram(name: "Test_Func_StructPointerElements", linkageName: "_Z31Test_Func_StructPointerElements21StructPointerElements", scope: !4, file: !4, line: 501, type: !675, scopeLine: 501, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !683)
!675 = !DISubroutineType(types: !676)
!676 = !{null, !677}
!677 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructPointerElements", file: !4, line: 93, size: 128, flags: DIFlagTypePassByValue, elements: !678, identifier: "_ZTS21StructPointerElements")
!678 = !{!679, !680}
!679 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !677, file: !4, line: 94, baseType: !630, size: 64)
!680 = !DIDerivedType(tag: DW_TAG_member, name: "Element1", scope: !677, file: !4, line: 95, baseType: !681, size: 64, offset: 64)
!681 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !682, size: 64)
!682 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!683 = !{!684}
!684 = !DILocalVariable(arg: 1, scope: !674, file: !4, line: 501, type: !677)
!685 = distinct !DILifetime(object: !684, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructPointerElements)))
!686 = !DILocation(line: 501, column: 70, scope: !674)
!687 = !DILocation(line: 501, column: 73, scope: !674)
!688 = distinct !DISubprogram(name: "Test_Kern_StructPointerElements", linkageName: "_Z31Test_Kern_StructPointerElements21StructPointerElements", scope: !4, file: !4, line: 506, type: !675, scopeLine: 506, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !689)
!689 = !{!690}
!690 = !DILocalVariable(arg: 1, scope: !688, file: !4, line: 506, type: !677)
!691 = distinct !DILifetime(object: !690, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructPointerElements)))
!692 = !DILocation(line: 506, column: 70, scope: !688)
!693 = !DILocation(line: 506, column: 73, scope: !688)
!694 = distinct !DISubprogram(name: "Test_Func_ParamRegLimitExpandedStruct", linkageName: "_Z37Test_Func_ParamRegLimitExpandedStructlllllli22StructMultipleElements", scope: !4, file: !4, line: 513, type: !695, scopeLine: 513, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !701)
!695 = !DISubroutineType(types: !696)
!696 = !{null, !18, !18, !18, !18, !18, !18, !14, !697}
!697 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "StructMultipleElements", file: !4, line: 98, size: 128, flags: DIFlagTypePassByValue, elements: !698, identifier: "_ZTS22StructMultipleElements")
!698 = !{!699, !700}
!699 = !DIDerivedType(tag: DW_TAG_member, name: "Element0", scope: !697, file: !4, line: 99, baseType: !14, size: 32)
!700 = !DIDerivedType(tag: DW_TAG_member, name: "Element1", scope: !697, file: !4, line: 100, baseType: !18, size: 64, offset: 64)
!701 = !{!702, !703, !704, !705, !706, !707, !708, !709}
!702 = !DILocalVariable(arg: 1, scope: !694, file: !4, line: 513, type: !18)
!703 = !DILocalVariable(arg: 2, scope: !694, file: !4, line: 513, type: !18)
!704 = !DILocalVariable(arg: 3, scope: !694, file: !4, line: 513, type: !18)
!705 = !DILocalVariable(arg: 4, scope: !694, file: !4, line: 513, type: !18)
!706 = !DILocalVariable(arg: 5, scope: !694, file: !4, line: 513, type: !18)
!707 = !DILocalVariable(arg: 6, scope: !694, file: !4, line: 513, type: !18)
!708 = !DILocalVariable(arg: 7, scope: !694, file: !4, line: 513, type: !14)
!709 = !DILocalVariable(arg: 8, scope: !694, file: !4, line: 513, type: !697)
!710 = distinct !DILifetime(object: !702, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!711 = !DILocation(line: 513, column: 62, scope: !694)
!712 = distinct !DILifetime(object: !703, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!713 = !DILocation(line: 513, column: 71, scope: !694)
!714 = distinct !DILifetime(object: !704, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!715 = !DILocation(line: 513, column: 80, scope: !694)
!716 = distinct !DILifetime(object: !705, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!717 = !DILocation(line: 513, column: 89, scope: !694)
!718 = distinct !DILifetime(object: !706, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!719 = !DILocation(line: 513, column: 98, scope: !694)
!720 = distinct !DILifetime(object: !707, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!721 = !DILocation(line: 513, column: 107, scope: !694)
!722 = distinct !DILifetime(object: !708, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!723 = !DILocation(line: 513, column: 116, scope: !694)
!724 = distinct !DILifetime(object: !709, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructMultipleElements)))
!725 = !DILocation(line: 513, column: 140, scope: !694)
!726 = !DILocation(line: 513, column: 143, scope: !694)
!727 = distinct !DISubprogram(name: "Test_Kern_ParamRegLimitExpandedStruct", linkageName: "_Z37Test_Kern_ParamRegLimitExpandedStructlllllli22StructMultipleElements", scope: !4, file: !4, line: 518, type: !695, scopeLine: 518, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !728)
!728 = !{!729, !730, !731, !732, !733, !734, !735, !736}
!729 = !DILocalVariable(arg: 1, scope: !727, file: !4, line: 518, type: !18)
!730 = !DILocalVariable(arg: 2, scope: !727, file: !4, line: 518, type: !18)
!731 = !DILocalVariable(arg: 3, scope: !727, file: !4, line: 518, type: !18)
!732 = !DILocalVariable(arg: 4, scope: !727, file: !4, line: 518, type: !18)
!733 = !DILocalVariable(arg: 5, scope: !727, file: !4, line: 518, type: !18)
!734 = !DILocalVariable(arg: 6, scope: !727, file: !4, line: 518, type: !18)
!735 = !DILocalVariable(arg: 7, scope: !727, file: !4, line: 518, type: !14)
!736 = !DILocalVariable(arg: 8, scope: !727, file: !4, line: 518, type: !697)
!737 = distinct !DILifetime(object: !729, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!738 = !DILocation(line: 518, column: 62, scope: !727)
!739 = distinct !DILifetime(object: !730, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!740 = !DILocation(line: 518, column: 71, scope: !727)
!741 = distinct !DILifetime(object: !731, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!742 = !DILocation(line: 518, column: 80, scope: !727)
!743 = distinct !DILifetime(object: !732, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!744 = !DILocation(line: 518, column: 89, scope: !727)
!745 = distinct !DILifetime(object: !733, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!746 = !DILocation(line: 518, column: 98, scope: !727)
!747 = distinct !DILifetime(object: !734, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!748 = !DILocation(line: 518, column: 107, scope: !727)
!749 = distinct !DILifetime(object: !735, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!750 = !DILocation(line: 518, column: 116, scope: !727)
!751 = distinct !DILifetime(object: !736, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructMultipleElements)))
!752 = !DILocation(line: 518, column: 140, scope: !727)
!753 = !DILocation(line: 518, column: 143, scope: !727)
!754 = distinct !DISubprogram(name: "Test_Func_ParamRegLimitUnexpandedStruct", linkageName: "_Z39Test_Func_ParamRegLimitUnexpandedStructlllllll22StructMultipleElements", scope: !4, file: !4, line: 522, type: !755, scopeLine: 522, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !757)
!755 = !DISubroutineType(types: !756)
!756 = !{null, !18, !18, !18, !18, !18, !18, !18, !697}
!757 = !{!758, !759, !760, !761, !762, !763, !764, !765}
!758 = !DILocalVariable(arg: 1, scope: !754, file: !4, line: 522, type: !18)
!759 = !DILocalVariable(arg: 2, scope: !754, file: !4, line: 522, type: !18)
!760 = !DILocalVariable(arg: 3, scope: !754, file: !4, line: 522, type: !18)
!761 = !DILocalVariable(arg: 4, scope: !754, file: !4, line: 522, type: !18)
!762 = !DILocalVariable(arg: 5, scope: !754, file: !4, line: 522, type: !18)
!763 = !DILocalVariable(arg: 6, scope: !754, file: !4, line: 522, type: !18)
!764 = !DILocalVariable(arg: 7, scope: !754, file: !4, line: 522, type: !18)
!765 = !DILocalVariable(arg: 8, scope: !754, file: !4, line: 522, type: !697)
!766 = distinct !DILifetime(object: !758, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!767 = !DILocation(line: 522, column: 64, scope: !754)
!768 = distinct !DILifetime(object: !759, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!769 = !DILocation(line: 522, column: 73, scope: !754)
!770 = distinct !DILifetime(object: !760, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!771 = !DILocation(line: 522, column: 82, scope: !754)
!772 = distinct !DILifetime(object: !761, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!773 = !DILocation(line: 522, column: 91, scope: !754)
!774 = distinct !DILifetime(object: !762, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!775 = !DILocation(line: 522, column: 100, scope: !754)
!776 = distinct !DILifetime(object: !763, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!777 = !DILocation(line: 522, column: 109, scope: !754)
!778 = distinct !DILifetime(object: !764, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!779 = !DILocation(line: 522, column: 118, scope: !754)
!780 = distinct !DILifetime(object: !765, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructMultipleElements)))
!781 = !DILocation(line: 522, column: 142, scope: !754)
!782 = !DILocation(line: 522, column: 145, scope: !754)
!783 = distinct !DISubprogram(name: "Test_Kern_ParamRegLimitUnexpandedStruct", linkageName: "_Z39Test_Kern_ParamRegLimitUnexpandedStructlllllll22StructMultipleElements", scope: !4, file: !4, line: 527, type: !755, scopeLine: 527, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !784)
!784 = !{!785, !786, !787, !788, !789, !790, !791, !792}
!785 = !DILocalVariable(arg: 1, scope: !783, file: !4, line: 527, type: !18)
!786 = !DILocalVariable(arg: 2, scope: !783, file: !4, line: 527, type: !18)
!787 = !DILocalVariable(arg: 3, scope: !783, file: !4, line: 527, type: !18)
!788 = !DILocalVariable(arg: 4, scope: !783, file: !4, line: 527, type: !18)
!789 = !DILocalVariable(arg: 5, scope: !783, file: !4, line: 527, type: !18)
!790 = !DILocalVariable(arg: 6, scope: !783, file: !4, line: 527, type: !18)
!791 = !DILocalVariable(arg: 7, scope: !783, file: !4, line: 527, type: !18)
!792 = !DILocalVariable(arg: 8, scope: !783, file: !4, line: 527, type: !697)
!793 = distinct !DILifetime(object: !785, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!794 = !DILocation(line: 527, column: 64, scope: !783)
!795 = distinct !DILifetime(object: !786, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!796 = !DILocation(line: 527, column: 73, scope: !783)
!797 = distinct !DILifetime(object: !787, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!798 = !DILocation(line: 527, column: 82, scope: !783)
!799 = distinct !DILifetime(object: !788, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!800 = !DILocation(line: 527, column: 91, scope: !783)
!801 = distinct !DILifetime(object: !789, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!802 = !DILocation(line: 527, column: 100, scope: !783)
!803 = distinct !DILifetime(object: !790, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!804 = !DILocation(line: 527, column: 109, scope: !783)
!805 = distinct !DILifetime(object: !791, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i64)))
!806 = !DILocation(line: 527, column: 118, scope: !783)
!807 = distinct !DILifetime(object: !792, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.StructMultipleElements)))
!808 = !DILocation(line: 527, column: 142, scope: !783)
!809 = !DILocation(line: 527, column: 145, scope: !783)
