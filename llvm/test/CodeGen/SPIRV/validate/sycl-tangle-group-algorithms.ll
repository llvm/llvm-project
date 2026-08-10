; This is an excerpt from the SYCL end-to-end test suite, cleaned out from
; unrelevant details, that reproduced cases of invalid SPIR-V generation due
; to wrong types, deduced from the input LLVM IR. Namely, this test case covers
; cases of type mismatch when null pointer constant is used in different
; contexts and so with different pointee types, and intertwined
; load/store/function call LLVM IR input with bitcasts inserted between
; instruction uses.

; The only pass criterion is that spirv-val considers output valid.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.5-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.5 %}

%"nd_item" = type { i8 }
%struct.AssertHappened = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
%"range" = type { %"detail::array" }
%"detail::array" = type { [1 x i64] }
%class.anon = type { %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor" }
%"accessor" = type { %"detail::AccessorImplDevice", %union.anon }
%"detail::AccessorImplDevice" = type { %"range", %"range", %"range" }
%union.anon = type { ptr addrspace(1) }
%class.anon.6 = type { ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4) }
%"group" = type { %"range", %"range", %"range", %"range" }
%"item" = type { %"detail::AccessorImplDevice" }
%"item.22" = type { %"sd_ItemBase.23" }
%"sd_ItemBase.23" = type { %"range", %"range" }
%"tangle_group" = type { %"ss_sub_group_mask" }
%"ss_sub_group_mask" = type { i64, i64 }
%class.anon.8 = type { %"accessor", %"accessor", [8 x i8], %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor", %"accessor" }
%"vec.16" = type { %"struct.std::array.20" }
%"struct.std::array.20" = type { [4 x i32] }
%class.anon.15 = type { ptr addrspace(4), ptr addrspace(4), ptr addrspace(4) }
%class.anon.7 = type { ptr addrspace(4), ptr addrspace(4) }

@.str = private unnamed_addr addrspace(1) constant [21 x i8] c"bits_num <= max_bits\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [17 x i8] c"subgroupmask.hpp\00", align 1
@__PRETTY_FUNCTION1 = private unnamed_addr addrspace(1) constant [32 x i8] c"subgroup_mask(BitsType, size_t)\00", align 1
@.str.2 = private unnamed_addr addrspace(1) constant [15 x i8] c"bn <= max_bits\00", align 1
@__PRETTY_FUNCTION2 = private unnamed_addr addrspace(1) constant [52 x i8] c"BitsType subgroup_mask::valuable_bits(size_t) const\00", align 1
@__spirv_BuiltInSubgroupMaxSize = external dso_local addrspace(1) constant i32, align 4
@__spirv_BuiltInSubgroupLocalInvocationId = external dso_local addrspace(1) constant i32, align 4
@_ZSt6ignore = linkonce_odr dso_local addrspace(1) constant %"nd_item" undef, align 1
@__spirv_BuiltInNumWorkgroups = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalOffset = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalSize = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@SPIR_AssertHappenedMem = linkonce_odr dso_local addrspace(1) global %struct.AssertHappened zeroinitializer
@__spirv_BuiltInWorkgroupId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupSize = external dso_local addrspace(1) constant <3 x i64>, align 32


define weak_odr dso_local spir_kernel void @TestKernel(ptr addrspace(1) %_arg_TmpAcc, ptr byval(%"range") %_arg_TmpAcc1, ptr byval(%"range") %_arg_TmpAcc2, ptr byval(%"range") %_arg_TmpAcc3, ptr addrspace(1) align 1 %_arg_BarrierAcc, ptr byval(%"range") %_arg_BarrierAcc4, ptr byval(%"range") %_arg_BarrierAcc5, ptr byval(%"range") %_arg_BarrierAcc6, ptr addrspace(1) align 1 %_arg_BroadcastAcc, ptr byval(%"range") %_arg_BroadcastAcc7, ptr byval(%"range") %_arg_BroadcastAcc8, ptr byval(%"range") %_arg_BroadcastAcc9, ptr addrspace(1) align 1 %_arg_AnyAcc, ptr byval(%"range") %_arg_AnyAcc10, ptr byval(%"range") %_arg_AnyAcc11, ptr byval(%"range") %_arg_AnyAcc12, ptr addrspace(1) align 1 %_arg_AllAcc, ptr byval(%"range") %_arg_AllAcc13, ptr byval(%"range") %_arg_AllAcc14, ptr byval(%"range") %_arg_AllAcc15, ptr addrspace(1) align 1 %_arg_NoneAcc, ptr byval(%"range") %_arg_NoneAcc16, ptr byval(%"range") %_arg_NoneAcc17, ptr byval(%"range") %_arg_NoneAcc18, ptr addrspace(1) align 1 %_arg_ReduceAcc, ptr byval(%"range") %_arg_ReduceAcc19, ptr byval(%"range") %_arg_ReduceAcc20, ptr byval(%"range") %_arg_ReduceAcc21, ptr addrspace(1) align 1 %_arg_ExScanAcc, ptr byval(%"range") %_arg_ExScanAcc22, ptr byval(%"range") %_arg_ExScanAcc23, ptr byval(%"range") %_arg_ExScanAcc24, ptr addrspace(1) align 1 %_arg_IncScanAcc, ptr byval(%"range") %_arg_IncScanAcc25, ptr byval(%"range") %_arg_IncScanAcc26, ptr byval(%"range") %_arg_IncScanAcc27, ptr addrspace(1) align 1 %_arg_ShiftLeftAcc, ptr byval(%"range") %_arg_ShiftLeftAcc28, ptr byval(%"range") %_arg_ShiftLeftAcc29, ptr byval(%"range") %_arg_ShiftLeftAcc30, ptr addrspace(1) align 1 %_arg_ShiftRightAcc, ptr byval(%"range") %_arg_ShiftRightAcc31, ptr byval(%"range") %_arg_ShiftRightAcc32, ptr byval(%"range") %_arg_ShiftRightAcc33, ptr addrspace(1) align 1 %_arg_SelectAcc, ptr byval(%"range") %_arg_SelectAcc34, ptr byval(%"range") %_arg_SelectAcc35, ptr byval(%"range") %_arg_SelectAcc36, ptr addrspace(1) align 1 %_arg_PermuteXorAcc, ptr byval(%"range") %_arg_PermuteXorAcc37, ptr byval(%"range") %_arg_PermuteXorAcc38, ptr byval(%"range") %_arg_PermuteXorAcc39) {
entry:
  %_arg_TmpAcc.addr = alloca ptr addrspace(1)
  %_arg_BarrierAcc.addr = alloca ptr addrspace(1)
  %_arg_BroadcastAcc.addr = alloca ptr addrspace(1)
  %_arg_AnyAcc.addr = alloca ptr addrspace(1)
  %_arg_AllAcc.addr = alloca ptr addrspace(1)
  %_arg_NoneAcc.addr = alloca ptr addrspace(1)
  %_arg_ReduceAcc.addr = alloca ptr addrspace(1)
  %_arg_ExScanAcc.addr = alloca ptr addrspace(1)
  %_arg_IncScanAcc.addr = alloca ptr addrspace(1)
  %_arg_ShiftLeftAcc.addr = alloca ptr addrspace(1)
  %_arg_ShiftRightAcc.addr = alloca ptr addrspace(1)
  %_arg_SelectAcc.addr = alloca ptr addrspace(1)
  %_arg_PermuteXorAcc.addr = alloca ptr addrspace(1)
  %Kernel = alloca %class.anon
  %agg.tmp = alloca %"range"
  %agg.tmp41 = alloca %"range"
  %agg.tmp42 = alloca %"range"
  %agg.tmp44 = alloca %"range"
  %agg.tmp45 = alloca %"range"
  %agg.tmp46 = alloca %"range"
  %agg.tmp48 = alloca %"range"
  %agg.tmp49 = alloca %"range"
  %agg.tmp50 = alloca %"range"
  %agg.tmp52 = alloca %"range"
  %agg.tmp53 = alloca %"range"
  %agg.tmp54 = alloca %"range"
  %agg.tmp56 = alloca %"range"
  %agg.tmp57 = alloca %"range"
  %agg.tmp58 = alloca %"range"
  %agg.tmp60 = alloca %"range"
  %agg.tmp61 = alloca %"range"
  %agg.tmp62 = alloca %"range"
  %agg.tmp64 = alloca %"range"
  %agg.tmp65 = alloca %"range"
  %agg.tmp66 = alloca %"range"
  %agg.tmp68 = alloca %"range"
  %agg.tmp69 = alloca %"range"
  %agg.tmp70 = alloca %"range"
  %agg.tmp72 = alloca %"range"
  %agg.tmp73 = alloca %"range"
  %agg.tmp74 = alloca %"range"
  %agg.tmp76 = alloca %"range"
  %agg.tmp77 = alloca %"range"
  %agg.tmp78 = alloca %"range"
  %agg.tmp80 = alloca %"range"
  %agg.tmp81 = alloca %"range"
  %agg.tmp82 = alloca %"range"
  %agg.tmp84 = alloca %"range"
  %agg.tmp85 = alloca %"range"
  %agg.tmp86 = alloca %"range"
  %agg.tmp88 = alloca %"range"
  %agg.tmp89 = alloca %"range"
  %agg.tmp90 = alloca %"range"
  %agg.tmp91 = alloca %"nd_item", align 1
  %Kernel.ascast = addrspacecast ptr %Kernel to ptr addrspace(4)
  %agg.tmp91.ascast = addrspacecast ptr %agg.tmp91 to ptr addrspace(4)
  store ptr addrspace(1) %_arg_TmpAcc, ptr %_arg_TmpAcc.addr
  store ptr addrspace(1) %_arg_BarrierAcc, ptr %_arg_BarrierAcc.addr
  store ptr addrspace(1) %_arg_BroadcastAcc, ptr %_arg_BroadcastAcc.addr
  store ptr addrspace(1) %_arg_AnyAcc, ptr %_arg_AnyAcc.addr
  store ptr addrspace(1) %_arg_AllAcc, ptr %_arg_AllAcc.addr
  store ptr addrspace(1) %_arg_NoneAcc, ptr %_arg_NoneAcc.addr
  store ptr addrspace(1) %_arg_ReduceAcc, ptr %_arg_ReduceAcc.addr
  store ptr addrspace(1) %_arg_ExScanAcc, ptr %_arg_ExScanAcc.addr
  store ptr addrspace(1) %_arg_IncScanAcc, ptr %_arg_IncScanAcc.addr
  store ptr addrspace(1) %_arg_ShiftLeftAcc, ptr %_arg_ShiftLeftAcc.addr
  store ptr addrspace(1) %_arg_ShiftRightAcc, ptr %_arg_ShiftRightAcc.addr
  store ptr addrspace(1) %_arg_SelectAcc, ptr %_arg_SelectAcc.addr
  store ptr addrspace(1) %_arg_PermuteXorAcc, ptr %_arg_PermuteXorAcc.addr
  %TmpAcc1 = bitcast ptr addrspace(4) %Kernel.ascast to ptr addrspace(4)
  call spir_func void @Foo1(ptr addrspace(4) %TmpAcc1) 
  %BarrierAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 1
  call spir_func void @Foo2(ptr addrspace(4) %BarrierAcc) 
  %BroadcastAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 2
  call spir_func void @Foo2(ptr addrspace(4) %BroadcastAcc) 
  %AnyAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 3
  call spir_func void @Foo2(ptr addrspace(4) %AnyAcc) 
  %AllAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 4
  call spir_func void @Foo2(ptr addrspace(4) %AllAcc) 
  %NoneAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 5
  call spir_func void @Foo2(ptr addrspace(4) %NoneAcc) 
  %ReduceAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 6
  call spir_func void @Foo2(ptr addrspace(4) %ReduceAcc) 
  %ExScanAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 7
  call spir_func void @Foo2(ptr addrspace(4) %ExScanAcc) 
  %IncScanAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 8
  call spir_func void @Foo2(ptr addrspace(4) %IncScanAcc) 
  %ShiftLeftAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 9
  call spir_func void @Foo2(ptr addrspace(4) %ShiftLeftAcc) 
  %ShiftRightAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 10
  call spir_func void @Foo2(ptr addrspace(4) %ShiftRightAcc) 
  %SelectAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 11
  call spir_func void @Foo2(ptr addrspace(4) %SelectAcc) 
  %PermuteXorAcc = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 12
  call spir_func void @Foo2(ptr addrspace(4) %PermuteXorAcc) 
  %TmpAcc402 = bitcast ptr addrspace(4) %Kernel.ascast to ptr addrspace(4)
  %0 = load ptr addrspace(1), ptr %_arg_TmpAcc.addr
  call spir_func void @Foo3(ptr addrspace(4) %TmpAcc402, ptr addrspace(1) %0, ptr byval(%"range") %agg.tmp, ptr byval(%"range") %agg.tmp41, ptr byval(%"range") %agg.tmp42) 
  %BarrierAcc43 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 1
  %1 = load ptr addrspace(1), ptr %_arg_BarrierAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %BarrierAcc43, ptr addrspace(1) %1, ptr byval(%"range") %agg.tmp44, ptr byval(%"range") %agg.tmp45, ptr byval(%"range") %agg.tmp46) 
  %BroadcastAcc47 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 2
  %2 = load ptr addrspace(1), ptr %_arg_BroadcastAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %BroadcastAcc47, ptr addrspace(1) %2, ptr byval(%"range") %agg.tmp48, ptr byval(%"range") %agg.tmp49, ptr byval(%"range") %agg.tmp50) 
  %AnyAcc51 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 3
  %3 = load ptr addrspace(1), ptr %_arg_AnyAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %AnyAcc51, ptr addrspace(1) %3, ptr byval(%"range") %agg.tmp52, ptr byval(%"range") %agg.tmp53, ptr byval(%"range") %agg.tmp54) 
  %AllAcc55 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 4
  %4 = load ptr addrspace(1), ptr %_arg_AllAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %AllAcc55, ptr addrspace(1) %4, ptr byval(%"range") %agg.tmp56, ptr byval(%"range") %agg.tmp57, ptr byval(%"range") %agg.tmp58) 
  %NoneAcc59 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 5
  %5 = load ptr addrspace(1), ptr %_arg_NoneAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %NoneAcc59, ptr addrspace(1) %5, ptr byval(%"range") %agg.tmp60, ptr byval(%"range") %agg.tmp61, ptr byval(%"range") %agg.tmp62) 
  %ReduceAcc63 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 6
  %6 = load ptr addrspace(1), ptr %_arg_ReduceAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %ReduceAcc63, ptr addrspace(1) %6, ptr byval(%"range") %agg.tmp64, ptr byval(%"range") %agg.tmp65, ptr byval(%"range") %agg.tmp66) 
  %ExScanAcc67 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 7
  %7 = load ptr addrspace(1), ptr %_arg_ExScanAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %ExScanAcc67, ptr addrspace(1) %7, ptr byval(%"range") %agg.tmp68, ptr byval(%"range") %agg.tmp69, ptr byval(%"range") %agg.tmp70) 
  %IncScanAcc71 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 8
  %8 = load ptr addrspace(1), ptr %_arg_IncScanAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %IncScanAcc71, ptr addrspace(1) %8, ptr byval(%"range") %agg.tmp72, ptr byval(%"range") %agg.tmp73, ptr byval(%"range") %agg.tmp74) 
  %ShiftLeftAcc75 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 9
  %9 = load ptr addrspace(1), ptr %_arg_ShiftLeftAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %ShiftLeftAcc75, ptr addrspace(1) %9, ptr byval(%"range") %agg.tmp76, ptr byval(%"range") %agg.tmp77, ptr byval(%"range") %agg.tmp78) 
  %ShiftRightAcc79 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 10
  %10 = load ptr addrspace(1), ptr %_arg_ShiftRightAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %ShiftRightAcc79, ptr addrspace(1) %10, ptr byval(%"range") %agg.tmp80, ptr byval(%"range") %agg.tmp81, ptr byval(%"range") %agg.tmp82) 
  %SelectAcc83 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 11
  %11 = load ptr addrspace(1), ptr %_arg_SelectAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %SelectAcc83, ptr addrspace(1) %11, ptr byval(%"range") %agg.tmp84, ptr byval(%"range") %agg.tmp85, ptr byval(%"range") %agg.tmp86) 
  %PermuteXorAcc87 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %Kernel.ascast, i32 0, i32 12
  %12 = load ptr addrspace(1), ptr %_arg_PermuteXorAcc.addr
  call spir_func void @Foo4(ptr addrspace(4) %PermuteXorAcc87, ptr addrspace(1) %12, ptr byval(%"range") %agg.tmp88, ptr byval(%"range") %agg.tmp89, ptr byval(%"range") %agg.tmp90) 
  %call = call spir_func ptr addrspace(4) @Foo5() 
  call spir_func void @Foo6(ptr addrspace(4) dead_on_unwind writable sret(%"nd_item") align 1 %agg.tmp91.ascast, ptr addrspace(4) %call) 
  call spir_func void @Foo22(ptr addrspace(4) %Kernel.ascast, ptr byval(%"nd_item") align 1 %agg.tmp91) 
  ret void
}

define internal spir_func void @Foo1(ptr addrspace(4) %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"range"
  %agg.tmp2 = alloca %"range"
  %agg.tmp3 = alloca %"range"
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %agg.tmp2.ascast = addrspacecast ptr %agg.tmp2 to ptr addrspace(4)
  %agg.tmp3.ascast = addrspacecast ptr %agg.tmp3 to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  call void @llvm.memset.p0.i64(ptr %agg.tmp, i8 0, i64 8, i1 false)
  call spir_func void @Foo11(ptr addrspace(4) %agg.tmp.ascast) 
  call spir_func void @Foo12(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp2.ascast) 
  call spir_func void @Foo12(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp3.ascast) 
  call spir_func void @Foo10(ptr addrspace(4) %impl1, ptr byval(%"range") %agg.tmp, ptr byval(%"range") %agg.tmp2, ptr byval(%"range") %agg.tmp3) 
  ret void
}


define internal spir_func void @Foo2(ptr addrspace(4) %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"range"
  %agg.tmp2 = alloca %"range"
  %agg.tmp3 = alloca %"range"
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %agg.tmp2.ascast = addrspacecast ptr %agg.tmp2 to ptr addrspace(4)
  %agg.tmp3.ascast = addrspacecast ptr %agg.tmp3 to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  call void @llvm.memset.p0.i64(ptr %agg.tmp, i8 0, i64 8, i1 false)
  call spir_func void @Foo11(ptr addrspace(4) %agg.tmp.ascast) 
  call spir_func void @Foo12(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp2.ascast) 
  call spir_func void @Foo12(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp3.ascast) 
  call spir_func void @Foo10(ptr addrspace(4) %impl1, ptr byval(%"range") %agg.tmp, ptr byval(%"range") %agg.tmp2, ptr byval(%"range") %agg.tmp3) 
  ret void
}




define internal spir_func void @Foo3(ptr addrspace(4) %this, ptr addrspace(1) %Ptr, ptr byval(%"range") %AccessRange, ptr byval(%"range") %MemRange, ptr byval(%"range") %Offset) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %Ptr.addr = alloca ptr addrspace(1)
  %ref.tmp = alloca %class.anon.6
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(1) %Ptr, ptr %Ptr.addr
  %AccessRange.ascast = addrspacecast ptr %AccessRange to ptr addrspace(4)
  %MemRange.ascast = addrspacecast ptr %MemRange to ptr addrspace(4)
  %Offset.ascast = addrspacecast ptr %Offset to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load ptr addrspace(1), ptr %Ptr.addr
  %1 = getelementptr inbounds nuw %"accessor", ptr addrspace(4) %this1, i32 0, i32 1
  store ptr addrspace(1) %0, ptr addrspace(4) %1
  %2 = bitcast ptr %ref.tmp to ptr
  store ptr addrspace(4) %this1, ptr %2
  %Offset2 = getelementptr inbounds %class.anon.6, ptr %ref.tmp, i32 0, i32 1
  store ptr addrspace(4) %Offset.ascast, ptr %Offset2
  %AccessRange3 = getelementptr inbounds %class.anon.6, ptr %ref.tmp, i32 0, i32 2
  store ptr addrspace(4) %AccessRange.ascast, ptr %AccessRange3
  %MemRange4 = getelementptr inbounds %class.anon.6, ptr %ref.tmp, i32 0, i32 3
  store ptr addrspace(4) %MemRange.ascast, ptr %MemRange4
  call spir_func void @Foo13(ptr addrspace(4) %ref.tmp.ascast) 
  %call = call spir_func i64 @Foo21(ptr addrspace(4) %this1) 
  %3 = getelementptr inbounds nuw %"accessor", ptr addrspace(4) %this1, i32 0, i32 1
  %4 = load ptr addrspace(1), ptr addrspace(4) %3
  %add.ptr = getelementptr inbounds nuw i64, ptr addrspace(1) %4, i64 %call
  store ptr addrspace(1) %add.ptr, ptr addrspace(4) %3
  ret void
}


define internal spir_func void @Foo4(ptr addrspace(4) %this, ptr addrspace(1) %Ptr, ptr byval(%"range") %AccessRange, ptr byval(%"range") %MemRange, ptr byval(%"range") %Offset) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %Ptr.addr = alloca ptr addrspace(1)
  %ref.tmp = alloca %class.anon.6
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(1) %Ptr, ptr %Ptr.addr
  %AccessRange.ascast = addrspacecast ptr %AccessRange to ptr addrspace(4)
  %MemRange.ascast = addrspacecast ptr %MemRange to ptr addrspace(4)
  %Offset.ascast = addrspacecast ptr %Offset to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load ptr addrspace(1), ptr %Ptr.addr
  %1 = getelementptr inbounds nuw %"accessor", ptr addrspace(4) %this1, i32 0, i32 1
  store ptr addrspace(1) %0, ptr addrspace(4) %1
  %2 = bitcast ptr %ref.tmp to ptr
  store ptr addrspace(4) %this1, ptr %2
  %Offset2 = getelementptr inbounds %class.anon.6, ptr %ref.tmp, i32 0, i32 1
  store ptr addrspace(4) %Offset.ascast, ptr %Offset2
  %AccessRange3 = getelementptr inbounds %class.anon.6, ptr %ref.tmp, i32 0, i32 2
  store ptr addrspace(4) %AccessRange.ascast, ptr %AccessRange3
  %MemRange4 = getelementptr inbounds %class.anon.6, ptr %ref.tmp, i32 0, i32 3
  store ptr addrspace(4) %MemRange.ascast, ptr %MemRange4
  call spir_func void @Foo30(ptr addrspace(4) %ref.tmp.ascast) 
  %call = call spir_func i64 @Foo32(ptr addrspace(4) %this1) 
  %3 = getelementptr inbounds nuw %"accessor", ptr addrspace(4) %this1, i32 0, i32 1
  %4 = load ptr addrspace(1), ptr addrspace(4) %3
  %add.ptr = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 %call
  store ptr addrspace(1) %add.ptr, ptr addrspace(4) %3
  ret void
}


define internal spir_func ptr addrspace(4) @Foo5() {
entry:
  %retval = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  ret ptr addrspace(4) null
}


define internal spir_func void @Foo6(ptr addrspace(4) dead_on_unwind noalias writable sret(%"nd_item") align 1 %agg.result, ptr addrspace(4) %0) {
entry:
  %.addr = alloca ptr addrspace(4)
  %GlobalSize = alloca %"range"
  %LocalSize = alloca %"range"
  %GroupRange = alloca %"range"
  %GroupId = alloca %"range"
  %GlobalId = alloca %"range"
  %LocalId = alloca %"range"
  %GlobalOffset = alloca %"range"
  %Group = alloca %"group"
  %GlobalItem = alloca %"item"
  %LocalItem = alloca %"item.22"
  %cleanup.dest.slot = alloca i32, align 4
  %GlobalSize.ascast = addrspacecast ptr %GlobalSize to ptr addrspace(4)
  %LocalSize.ascast = addrspacecast ptr %LocalSize to ptr addrspace(4)
  %GroupRange.ascast = addrspacecast ptr %GroupRange to ptr addrspace(4)
  %GroupId.ascast = addrspacecast ptr %GroupId to ptr addrspace(4)
  %GlobalId.ascast = addrspacecast ptr %GlobalId to ptr addrspace(4)
  %LocalId.ascast = addrspacecast ptr %LocalId to ptr addrspace(4)
  %GlobalOffset.ascast = addrspacecast ptr %GlobalOffset to ptr addrspace(4)
  %Group.ascast = addrspacecast ptr %Group to ptr addrspace(4)
  %GlobalItem.ascast = addrspacecast ptr %GlobalItem to ptr addrspace(4)
  %LocalItem.ascast = addrspacecast ptr %LocalItem to ptr addrspace(4)
  store ptr addrspace(4) %0, ptr %.addr
  call spir_func void @Foo7(ptr addrspace(4) dead_on_unwind writable sret(%"range") %GlobalSize.ascast) 
  call spir_func void @Init1(ptr addrspace(4) dead_on_unwind writable sret(%"range") %LocalSize.ascast) 
  call spir_func void @Init2(ptr addrspace(4) dead_on_unwind writable sret(%"range") %GroupRange.ascast) 
  call spir_func void @Init3(ptr addrspace(4) dead_on_unwind writable sret(%"range") %GroupId.ascast) 
  call spir_func void @Init6(ptr addrspace(4) dead_on_unwind writable sret(%"range") %GlobalId.ascast) 
  call spir_func void @Init4(ptr addrspace(4) dead_on_unwind writable sret(%"range") %LocalId.ascast) 
  call spir_func void @Init5(ptr addrspace(4) dead_on_unwind writable sret(%"range") %GlobalOffset.ascast) 
  call spir_func void @Foo23(ptr addrspace(4) dead_on_unwind writable sret(%"group") %Group.ascast, ptr addrspace(4) %GlobalSize.ascast, ptr addrspace(4) %LocalSize.ascast, ptr addrspace(4) %GroupRange.ascast, ptr addrspace(4) %GroupId.ascast) 
  call spir_func void @Foo24(ptr addrspace(4) dead_on_unwind writable sret(%"item") %GlobalItem.ascast, ptr addrspace(4) %GlobalSize.ascast, ptr addrspace(4) %GlobalId.ascast, ptr addrspace(4) %GlobalOffset.ascast) 
  call spir_func void @Foo25(ptr addrspace(4) dead_on_unwind writable sret(%"item.22") %LocalItem.ascast, ptr addrspace(4) %LocalSize.ascast, ptr addrspace(4) %LocalId.ascast) 
  call spir_func void @Foo26(ptr addrspace(4) dead_on_unwind writable sret(%"nd_item") align 1 %agg.result, ptr addrspace(4) %GlobalItem.ascast, ptr addrspace(4) %LocalItem.ascast, ptr addrspace(4) %Group.ascast) 
  ret void
}


define internal spir_func void @Foo22(ptr addrspace(4) %this, ptr byval(%"nd_item") align 1 %item) {
entry:
  %this.addr.i76 = alloca ptr addrspace(4)
  %WI.addr.i = alloca i64
  %TangleLeader.addr.i = alloca i64
  %TangleSize.addr.i = alloca i64
  %agg.tmp.i = alloca %"range"
  %agg.tmp2.i = alloca %"tangle_group"
  %Visible.i = alloca i64
  %Other.i = alloca i64
  %agg.tmp5.i = alloca %"range"
  %agg.tmp8.i = alloca %"range"
  %OriginalLID.i = alloca i32, align 4
  %LID.i = alloca i32, align 4
  %BroadcastResult.i = alloca i32, align 4
  %agg.tmp12.i = alloca %"tangle_group"
  %agg.tmp15.i = alloca %"range"
  %AnyResult.i = alloca i8, align 1
  %agg.tmp18.i = alloca %"tangle_group"
  %agg.tmp24.i = alloca %"range"
  %AllResult.i = alloca i8, align 1
  %agg.tmp27.i = alloca %"tangle_group"
  %agg.tmp35.i = alloca %"range"
  %NoneResult.i = alloca i8, align 1
  %agg.tmp38.i = alloca %"tangle_group"
  %agg.tmp46.i = alloca %"range"
  %ReduceResult.i = alloca i32, align 4
  %agg.tmp49.i = alloca %"tangle_group"
  %agg.tmp50.i = alloca %"nd_item", align 1
  %agg.tmp54.i = alloca %"range"
  %ExScanResult.i = alloca i32, align 4
  %agg.tmp57.i = alloca %"tangle_group"
  %agg.tmp58.i = alloca %"nd_item", align 1
  %agg.tmp61.i = alloca %"range"
  %IncScanResult.i = alloca i32, align 4
  %agg.tmp64.i = alloca %"tangle_group"
  %agg.tmp65.i = alloca %"nd_item", align 1
  %agg.tmp69.i = alloca %"range"
  %ShiftLeftResult.i = alloca i32, align 4
  %agg.tmp72.i = alloca %"tangle_group"
  %agg.tmp79.i = alloca %"range"
  %ShiftRightResult.i = alloca i32, align 4
  %agg.tmp82.i = alloca %"tangle_group"
  %agg.tmp88.i = alloca %"range"
  %SelectResult.i = alloca i32, align 4
  %agg.tmp91.i = alloca %"tangle_group"
  %agg.tmp92.i = alloca %"range"
  %ref.tmp.i = alloca %"range"
  %ref.tmp93.i = alloca %"range"
  %ref.tmp94.i = alloca i32, align 4
  %agg.tmp100.i = alloca %"range"
  %PermuteXorResult.i = alloca i32, align 4
  %agg.tmp103.i = alloca %"tangle_group"
  %agg.tmp106.i = alloca %"range"
  %agg.tmp18.ascast.ascast75 = alloca %"nd_item"
  %agg.tmp17.ascast.ascast74 = alloca %"tangle_group"
  %retval.i66 = alloca i64
  %this.addr.i67 = alloca ptr addrspace(4)
  %Result.i68 = alloca i64
  %retval.i58 = alloca i64
  %this.addr.i59 = alloca ptr addrspace(4)
  %Result.i60 = alloca i64
  %retval.i50 = alloca i64
  %this.addr.i51 = alloca ptr addrspace(4)
  %Result.i52 = alloca i64
  %retval.i42 = alloca i64
  %this.addr.i43 = alloca ptr addrspace(4)
  %Result.i44 = alloca i64
  %retval.i = alloca i64
  %this.addr.i = alloca ptr addrspace(4)
  %Result.i = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %WI = alloca %"range"
  %SG = alloca %"nd_item", align 1
  %BranchBody = alloca %class.anon.8
  %ref.tmp = alloca %"range"
  %ref.tmp15 = alloca i32, align 4
  %Tangle = alloca %"tangle_group"
  %agg.tmp = alloca %"nd_item", align 1
  %TangleLeader = alloca i64
  %TangleSize = alloca i64
  %IsMember = alloca %"nd_item", align 1
  %agg.tmp17 = alloca %"tangle_group"
  %agg.tmp18 = alloca %"nd_item", align 1
  %ref.tmp19 = alloca %"range"
  %ref.tmp20 = alloca i32, align 4
  %Tangle24 = alloca %"tangle_group"
  %agg.tmp25 = alloca %"nd_item", align 1
  %TangleLeader26 = alloca i64
  %TangleSize27 = alloca i64
  %IsMember28 = alloca %"nd_item", align 1
  %agg.tmp30 = alloca %"tangle_group"
  %agg.tmp31 = alloca %"nd_item", align 1
  %Tangle33 = alloca %"tangle_group"
  %agg.tmp34 = alloca %"nd_item", align 1
  %TangleLeader35 = alloca i64
  %TangleSize36 = alloca i64
  %IsMember37 = alloca %"nd_item", align 1
  %agg.tmp39 = alloca %"tangle_group"
  %agg.tmp40 = alloca %"nd_item", align 1
  %WI.ascast = addrspacecast ptr %WI to ptr addrspace(4)
  %SG.ascast = addrspacecast ptr %SG to ptr addrspace(4)
  %BranchBody.ascast = addrspacecast ptr %BranchBody to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %ref.tmp15.ascast = addrspacecast ptr %ref.tmp15 to ptr addrspace(4)
  %Tangle.ascast = addrspacecast ptr %Tangle to ptr addrspace(4)
  %IsMember.ascast = addrspacecast ptr %IsMember to ptr addrspace(4)
  %ref.tmp19.ascast = addrspacecast ptr %ref.tmp19 to ptr addrspace(4)
  %ref.tmp20.ascast = addrspacecast ptr %ref.tmp20 to ptr addrspace(4)
  %Tangle24.ascast = addrspacecast ptr %Tangle24 to ptr addrspace(4)
  %IsMember28.ascast = addrspacecast ptr %IsMember28 to ptr addrspace(4)
  %Tangle33.ascast = addrspacecast ptr %Tangle33 to ptr addrspace(4)
  %IsMember37.ascast = addrspacecast ptr %IsMember37 to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %item.ascast = addrspacecast ptr %item to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @Foo40(ptr addrspace(4) dead_on_unwind writable sret(%"range") %WI.ascast, ptr addrspace(4) align 1 %item.ascast) 
  call spir_func void @Foo41(ptr addrspace(4) dead_on_unwind writable sret(%"nd_item") align 1 %SG.ascast, ptr addrspace(4) align 1 %item.ascast) 
  %TmpAcc1 = bitcast ptr %BranchBody to ptr
  %TmpAcc22 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %BarrierAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 1
  %BarrierAcc3 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 1
  %0 = getelementptr inbounds i8, ptr addrspace(4) %BranchBody.ascast, i64 64
  %BroadcastAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 3
  %BroadcastAcc4 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 2
  %AnyAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 4
  %AnyAcc5 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 3
  %AllAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 5
  %AllAcc6 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 4
  %NoneAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 6
  %NoneAcc7 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 5
  %ReduceAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 7
  %ReduceAcc8 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 6
  %ExScanAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 8
  %ExScanAcc9 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 7
  %IncScanAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 9
  %IncScanAcc10 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 8
  %ShiftLeftAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 10
  %ShiftLeftAcc11 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 9
  %ShiftRightAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 11
  %ShiftRightAcc12 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 10
  %SelectAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 12
  %SelectAcc13 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 11
  %PermuteXorAcc = getelementptr inbounds %class.anon.8, ptr %BranchBody, i32 0, i32 13
  %PermuteXorAcc14 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 12
  store i32 4, ptr %ref.tmp15, align 4
  call spir_func void @Foo42(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) %WI.ascast, ptr addrspace(4) align 4 %ref.tmp15.ascast) 
  %retval.ascast.i69 = addrspacecast ptr %retval.i66 to ptr addrspace(4)
  store ptr addrspace(4) %ref.tmp.ascast, ptr %this.addr.i67
  %this1.i72 = load ptr addrspace(4), ptr %this.addr.i67
  %1 = load i64, ptr addrspace(4) %this1.i72
  store i64 %1, ptr %Result.i68
  %2 = load i64, ptr %Result.i68
  %tobool = icmp ne i64 %2, 0
  br i1 %tobool, label %if.then, label %if.else

if.else:                                          ; preds = %entry
  store i32 24, ptr %ref.tmp20, align 4
  call spir_func void @Foo42(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp19.ascast, ptr addrspace(4) %WI.ascast, ptr addrspace(4) align 4 %ref.tmp20.ascast) 
  %retval.ascast.i53 = addrspacecast ptr %retval.i50 to ptr addrspace(4)
  store ptr addrspace(4) %ref.tmp19.ascast, ptr %this.addr.i51
  %this1.i56 = load ptr addrspace(4), ptr %this.addr.i51
  %3 = load i64, ptr addrspace(4) %this1.i56
  store i64 %3, ptr %Result.i52
  %4 = load i64, ptr %Result.i52
  %tobool22 = icmp ne i64 %4, 0
  br i1 %tobool22, label %if.then23, label %if.else32

if.else32:                                        ; preds = %if.else
  call spir_func void @Foo43(ptr addrspace(4) dead_on_unwind writable sret(%"tangle_group") %Tangle33.ascast, ptr byval(%"nd_item") align 1 %agg.tmp34) 
  store i64 24, ptr %TangleLeader35
  store i64 8, ptr %TangleSize36
  %retval.ascast.i = addrspacecast ptr %retval.i to ptr addrspace(4)
  store ptr addrspace(4) %WI.ascast, ptr %this.addr.i
  %this1.i = load ptr addrspace(4), ptr %this.addr.i
  %5 = load i64, ptr addrspace(4) %this1.i
  store i64 %5, ptr %Result.i
  %6 = load i64, ptr %Result.i
  %7 = load i64, ptr %TangleLeader35
  %8 = load i64, ptr %TangleSize36
  call spir_func void @Foo69(ptr addrspace(4) %BranchBody.ascast, i64 %6, ptr byval(%"tangle_group") %agg.tmp39, i64 %7, i64 %8, ptr byval(%"nd_item") align 1 %agg.tmp40) 
  br label %if.end41

if.then23:                                        ; preds = %if.else
  call spir_func void @Foo43(ptr addrspace(4) dead_on_unwind writable sret(%"tangle_group") %Tangle24.ascast, ptr byval(%"nd_item") align 1 %agg.tmp25) 
  store i64 4, ptr %TangleLeader26
  store i64 20, ptr %TangleSize27
  %retval.ascast.i45 = addrspacecast ptr %retval.i42 to ptr addrspace(4)
  store ptr addrspace(4) %WI.ascast, ptr %this.addr.i43
  %this1.i48 = load ptr addrspace(4), ptr %this.addr.i43
  %9 = load i64, ptr addrspace(4) %this1.i48
  store i64 %9, ptr %Result.i44
  %10 = load i64, ptr %Result.i44
  %11 = load i64, ptr %TangleLeader26
  %12 = load i64, ptr %TangleSize27
  call spir_func void @Foo68(ptr addrspace(4) %BranchBody.ascast, i64 %10, ptr byval(%"tangle_group") %agg.tmp30, i64 %11, i64 %12, ptr byval(%"nd_item") align 1 %agg.tmp31) 
  br label %if.end41

if.then:                                          ; preds = %entry
  call spir_func void @Foo43(ptr addrspace(4) dead_on_unwind writable sret(%"tangle_group") %Tangle.ascast, ptr byval(%"nd_item") align 1 %agg.tmp) 
  store i64 0, ptr %TangleLeader
  store i64 4, ptr %TangleSize
  %retval.ascast.i61 = addrspacecast ptr %retval.i58 to ptr addrspace(4)
  store ptr addrspace(4) %WI.ascast, ptr %this.addr.i59
  %this1.i64 = load ptr addrspace(4), ptr %this.addr.i59
  %13 = load i64, ptr addrspace(4) %this1.i64
  store i64 %13, ptr %Result.i60
  %14 = load i64, ptr %Result.i60
  %15 = load i64, ptr %TangleLeader
  %16 = load i64, ptr %TangleSize
  %TangleSize.addr.ascast.i = addrspacecast ptr %TangleSize.addr.i to ptr addrspace(4)
  %agg.tmp.ascast.i = addrspacecast ptr %agg.tmp.i to ptr addrspace(4)
  %agg.tmp5.ascast.i = addrspacecast ptr %agg.tmp5.i to ptr addrspace(4)
  %agg.tmp8.ascast.i = addrspacecast ptr %agg.tmp8.i to ptr addrspace(4)
  %agg.tmp15.ascast.i = addrspacecast ptr %agg.tmp15.i to ptr addrspace(4)
  %agg.tmp24.ascast.i = addrspacecast ptr %agg.tmp24.i to ptr addrspace(4)
  %agg.tmp35.ascast.i = addrspacecast ptr %agg.tmp35.i to ptr addrspace(4)
  %agg.tmp46.ascast.i = addrspacecast ptr %agg.tmp46.i to ptr addrspace(4)
  %agg.tmp50.ascast.i = addrspacecast ptr %agg.tmp50.i to ptr addrspace(4)
  %agg.tmp54.ascast.i = addrspacecast ptr %agg.tmp54.i to ptr addrspace(4)
  %agg.tmp58.ascast.i = addrspacecast ptr %agg.tmp58.i to ptr addrspace(4)
  %agg.tmp61.ascast.i = addrspacecast ptr %agg.tmp61.i to ptr addrspace(4)
  %agg.tmp65.ascast.i = addrspacecast ptr %agg.tmp65.i to ptr addrspace(4)
  %agg.tmp69.ascast.i = addrspacecast ptr %agg.tmp69.i to ptr addrspace(4)
  %agg.tmp79.ascast.i = addrspacecast ptr %agg.tmp79.i to ptr addrspace(4)
  %agg.tmp88.ascast.i = addrspacecast ptr %agg.tmp88.i to ptr addrspace(4)
  %agg.tmp92.ascast.i = addrspacecast ptr %agg.tmp92.i to ptr addrspace(4)
  %ref.tmp.ascast.i = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  %ref.tmp93.ascast.i = addrspacecast ptr %ref.tmp93.i to ptr addrspace(4)
  %ref.tmp94.ascast.i = addrspacecast ptr %ref.tmp94.i to ptr addrspace(4)
  %agg.tmp100.ascast.i = addrspacecast ptr %agg.tmp100.i to ptr addrspace(4)
  %agg.tmp106.ascast.i = addrspacecast ptr %agg.tmp106.i to ptr addrspace(4)
  store ptr addrspace(4) %BranchBody.ascast, ptr %this.addr.i76
  store i64 %14, ptr %WI.addr.i
  %Tangle.ascast.i = addrspacecast ptr %agg.tmp17.ascast.ascast74 to ptr addrspace(4)
  store i64 %15, ptr %TangleLeader.addr.i
  store i64 %16, ptr %TangleSize.addr.i
  %IsMember.ascast.i = addrspacecast ptr %agg.tmp18.ascast.ascast75 to ptr addrspace(4)
  %this1.i78 = load ptr addrspace(4), ptr %this.addr.i76
  %17 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp.ascast.i, i64 %17) 
  %call.i = call spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %this1.i78, ptr byval(%"range") %agg.tmp.i) 
  store i64 1, ptr addrspace(4) %call.i
  call spir_func void @Foo75(ptr byval(%"tangle_group") %agg.tmp2.i, i32 1) 
  store i64 0, ptr %Visible.i
  store i64 0, ptr %Other.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %if.end.i, %if.then
  %18 = load i64, ptr %Other.i
  %cmp.i79 = icmp ult i64 %18, 32
  br i1 %cmp.i79, label %for.body.i, label %for.cond.cleanup.i

for.cond.cleanup.i:                               ; preds = %for.cond.i
  %19 = load i64, ptr %Visible.i
  %20 = load i64, ptr %TangleSize.addr.i
  %cmp7.i = icmp eq i64 %19, %20
  %BarrierAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 1
  %21 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp8.ascast.i, i64 %21) 
  %call9.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %BarrierAcc.i, ptr byval(%"range") %agg.tmp8.i) 
  %storedv.i = zext i1 %cmp7.i to i8
  store i8 %storedv.i, ptr addrspace(4) %call9.i, align 1
  %22 = getelementptr inbounds i8, ptr addrspace(4) %this1.i78, i64 64
  %call10.i = call spir_func i32 @Foo76(ptr addrspace(4) align 1 %22) 
  store i32 %call10.i, ptr %OriginalLID.i, align 4
  %call11.i = call spir_func i32 @Foo90(ptr addrspace(4) %Tangle.ascast.i) 
  store i32 %call11.i, ptr %LID.i, align 4
  %23 = load i32, ptr %OriginalLID.i, align 4
  %call13.i = call spir_func i32 @Foo91(ptr byval(%"tangle_group") %agg.tmp12.i, i32 %23, i32 0) 
  store i32 %call13.i, ptr %BroadcastResult.i, align 4
  %24 = load i32, ptr %BroadcastResult.i, align 4
  %conv.i = zext i32 %24 to i64
  %25 = load i64, ptr %TangleLeader.addr.i
  %cmp14.i = icmp eq i64 %conv.i, %25
  %BroadcastAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 3
  %26 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp15.ascast.i, i64 %26) 
  %call16.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %BroadcastAcc.i, ptr byval(%"range") %agg.tmp15.i) 
  %storedv17.i = zext i1 %cmp14.i to i8
  store i8 %storedv17.i, ptr addrspace(4) %call16.i, align 1
  %27 = load i32, ptr %LID.i, align 4
  %cmp19.i = icmp eq i32 %27, 0
  %call20.i = call spir_func zeroext i1 @Foo92(ptr byval(%"tangle_group") %agg.tmp18.i, i1 zeroext %cmp19.i) 
  %storedv21.i = zext i1 %call20.i to i8
  store i8 %storedv21.i, ptr %AnyResult.i, align 1
  %28 = load i8, ptr %AnyResult.i, align 1  
  %loadedv.i = trunc i8 %28 to i1
  %conv22.i = zext i1 %loadedv.i to i32
  %AnyAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 4
  %29 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp24.ascast.i, i64 %29) 
  %call25.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %AnyAcc.i, ptr byval(%"range") %agg.tmp24.i) 
  %storedv26.i = zext i1 %loadedv.i to i8
  store i8 %storedv26.i, ptr addrspace(4) %call25.i, align 1
  %30 = load i32, ptr %LID.i, align 4
  %conv28.i = zext i32 %30 to i64
  %31 = load i64, ptr %TangleSize.addr.i
  %cmp29.i = icmp ult i64 %conv28.i, %31
  %call30.i = call spir_func zeroext i1 @Foo67(ptr byval(%"tangle_group") %agg.tmp27.i, i1 zeroext %cmp29.i) 
  %storedv31.i = zext i1 %call30.i to i8
  store i8 %storedv31.i, ptr %AllResult.i, align 1
  %32 = load i8, ptr %AllResult.i, align 1  
  %loadedv32.i = trunc i8 %32 to i1
  %conv33.i = zext i1 %loadedv32.i to i32
  %AllAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 5
  %33 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp35.ascast.i, i64 %33) 
  %call36.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %AllAcc.i, ptr byval(%"range") %agg.tmp35.i) 
  %storedv37.i = zext i1 %loadedv32.i to i8
  store i8 %storedv37.i, ptr addrspace(4) %call36.i, align 1
  %34 = load i32, ptr %LID.i, align 4
  %conv39.i = zext i32 %34 to i64
  %35 = load i64, ptr %TangleSize.addr.i
  %cmp40.i = icmp uge i64 %conv39.i, %35
  %call41.i = call spir_func zeroext i1 @Foo65(ptr byval(%"tangle_group") %agg.tmp38.i, i1 zeroext %cmp40.i) 
  %storedv42.i = zext i1 %call41.i to i8
  store i8 %storedv42.i, ptr %NoneResult.i, align 1
  %36 = load i8, ptr %NoneResult.i, align 1  
  %loadedv43.i = trunc i8 %36 to i1
  %conv44.i = zext i1 %loadedv43.i to i32
  %NoneAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 6
  %37 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp46.ascast.i, i64 %37) 
  %call47.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %NoneAcc.i, ptr byval(%"range") %agg.tmp46.i) 
  %storedv48.i = zext i1 %loadedv43.i to i8
  store i8 %storedv48.i, ptr addrspace(4) %call47.i, align 1
  %call51.i = call spir_func i32 @Foo64(ptr byval(%"tangle_group") %agg.tmp49.i, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp50.i) 
  store i32 %call51.i, ptr %ReduceResult.i, align 4
  %38 = load i32, ptr %ReduceResult.i, align 4
  %conv52.i = zext i32 %38 to i64
  %39 = load i64, ptr %TangleSize.addr.i
  %cmp53.i = icmp eq i64 %conv52.i, %39
  %ReduceAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 7
  %40 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp54.ascast.i, i64 %40) 
  %call55.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ReduceAcc.i, ptr byval(%"range") %agg.tmp54.i) 
  %storedv56.i = zext i1 %cmp53.i to i8
  store i8 %storedv56.i, ptr addrspace(4) %call55.i, align 1
  %call59.i = call spir_func i32 @Foo63(ptr byval(%"tangle_group") %agg.tmp57.i, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp58.i) 
  store i32 %call59.i, ptr %ExScanResult.i, align 4
  %41 = load i32, ptr %ExScanResult.i, align 4
  %42 = load i32, ptr %LID.i, align 4
  %cmp60.i = icmp eq i32 %41, %42
  %ExScanAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 8
  %43 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp61.ascast.i, i64 %43) 
  %call62.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ExScanAcc.i, ptr byval(%"range") %agg.tmp61.i) 
  %storedv63.i = zext i1 %cmp60.i to i8
  store i8 %storedv63.i, ptr addrspace(4) %call62.i, align 1
  %call66.i = call spir_func i32 @Foo62(ptr byval(%"tangle_group") %agg.tmp64.i, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp65.i) 
  store i32 %call66.i, ptr %IncScanResult.i, align 4
  %44 = load i32, ptr %IncScanResult.i, align 4
  %45 = load i32, ptr %LID.i, align 4
  %add67.i = add i32 %45, 1
  %cmp68.i = icmp eq i32 %44, %add67.i
  %IncScanAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 9
  %46 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp69.ascast.i, i64 %46) 
  %call70.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %IncScanAcc.i, ptr byval(%"range") %agg.tmp69.i) 
  %storedv71.i = zext i1 %cmp68.i to i8
  store i8 %storedv71.i, ptr addrspace(4) %call70.i, align 1
  %47 = load i32, ptr %LID.i, align 4
  %call73.i = call spir_func i32 @Foo73(ptr byval(%"tangle_group") %agg.tmp72.i, i32 %47, i32 2) 
  store i32 %call73.i, ptr %ShiftLeftResult.i, align 4
  %48 = load i32, ptr %LID.i, align 4
  %add74.i = add i32 %48, 2
  %conv75.i = zext i32 %add74.i to i64
  %49 = load i64, ptr %TangleSize.addr.i
  %cmp76.i = icmp uge i64 %conv75.i, %49
  br i1 %cmp76.i, label %lor.end.i, label %lor.rhs.i

lor.rhs.i:                                        ; preds = %for.cond.cleanup.i
  %50 = load i32, ptr %ShiftLeftResult.i, align 4
  %51 = load i32, ptr %LID.i, align 4
  %add77.i = add i32 %51, 2
  %cmp78.i = icmp eq i32 %50, %add77.i
  br label %lor.end.i

lor.end.i:                                        ; preds = %lor.rhs.i, %for.cond.cleanup.i
  %52 = phi i1 [ true, %for.cond.cleanup.i ], [ %cmp78.i, %lor.rhs.i ]
  %ShiftLeftAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 10
  %53 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp79.ascast.i, i64 %53) 
  %call80.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ShiftLeftAcc.i, ptr byval(%"range") %agg.tmp79.i) 
  %storedv81.i = zext i1 %52 to i8
  store i8 %storedv81.i, ptr addrspace(4) %call80.i, align 1
  %54 = load i32, ptr %LID.i, align 4
  %call83.i = call spir_func i32 @Foo53(ptr byval(%"tangle_group") %agg.tmp82.i, i32 %54, i32 2) 
  store i32 %call83.i, ptr %ShiftRightResult.i, align 4
  %55 = load i32, ptr %LID.i, align 4
  %cmp84.i = icmp ult i32 %55, 2
  br i1 %cmp84.i, label %l1.exit, label %lor.rhs85.i

lor.rhs85.i:                                      ; preds = %lor.end.i
  %56 = load i32, ptr %ShiftRightResult.i, align 4
  %57 = load i32, ptr %LID.i, align 4
  %sub.i = sub i32 %57, 2
  %cmp86.i = icmp eq i32 %56, %sub.i
  br label %l1.exit

l1.exit: ; preds = %lor.rhs85.i, %lor.end.i
  %58 = phi i1 [ true, %lor.end.i ], [ %cmp86.i, %lor.rhs85.i ]
  %ShiftRightAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 11
  %59 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp88.ascast.i, i64 %59) 
  %call89.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ShiftRightAcc.i, ptr byval(%"range") %agg.tmp88.i) 
  %storedv90.i = zext i1 %58 to i8
  store i8 %storedv90.i, ptr addrspace(4) %call89.i, align 1
  %60 = load i32, ptr %LID.i, align 4
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp93.ascast.i, ptr addrspace(4) %Tangle.ascast.i) 
  store i32 2, ptr %ref.tmp94.i, align 4
  call spir_func void @Foo55(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast.i, ptr addrspace(4) %ref.tmp93.ascast.i, ptr addrspace(4) align 4 %ref.tmp94.ascast.i) 
  call spir_func void @Foo56(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp92.ascast.i, ptr addrspace(4) %ref.tmp.ascast.i, ptr addrspace(4) %TangleSize.addr.ascast.i) 
  %call95.i = call spir_func i32 @Foo57(ptr byval(%"tangle_group") %agg.tmp91.i, i32 %60, ptr byval(%"range") %agg.tmp92.i) 
  store i32 %call95.i, ptr %SelectResult.i, align 4
  %61 = load i32, ptr %SelectResult.i, align 4
  %conv96.i = zext i32 %61 to i64
  %62 = load i32, ptr %LID.i, align 4
  %add97.i = add i32 %62, 2
  %conv98.i = zext i32 %add97.i to i64
  %63 = load i64, ptr %TangleSize.addr.i
  %rem.i = urem i64 %conv98.i, %63
  %cmp99.i = icmp eq i64 %conv96.i, %rem.i
  %SelectAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 12
  %64 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp100.ascast.i, i64 %64) 
  %call101.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %SelectAcc.i, ptr byval(%"range") %agg.tmp100.i) 
  %storedv102.i = zext i1 %cmp99.i to i8
  store i8 %storedv102.i, ptr addrspace(4) %call101.i, align 1
  %65 = load i32, ptr %LID.i, align 4
  %call104.i = call spir_func i32 @Foo58(ptr byval(%"tangle_group") %agg.tmp103.i, i32 %65, i32 2) 
  store i32 %call104.i, ptr %PermuteXorResult.i, align 4
  %66 = load i32, ptr %PermuteXorResult.i, align 4
  %67 = load i32, ptr %LID.i, align 4
  %xor.i = xor i32 %67, 2
  %cmp105.i = icmp eq i32 %66, %xor.i
  %PermuteXorAcc.i = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1.i78, i32 0, i32 13
  %68 = load i64, ptr %WI.addr.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp106.ascast.i, i64 %68) 
  %call107.i = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %PermuteXorAcc.i, ptr byval(%"range") %agg.tmp106.i) 
  %storedv108.i = zext i1 %cmp105.i to i8
  store i8 %storedv108.i, ptr addrspace(4) %call107.i, align 1
  br label %if.end41

if.end41:                                         ; preds = %if.then23, %if.else32, %l1.exit
  ret void

for.body.i:                                       ; preds = %for.cond.i
  %69 = load i64, ptr %Other.i
  %call3.i = call spir_func zeroext i1 @Foo71(ptr addrspace(4) align 1 %IsMember.ascast.i, i64 %69) 
  br i1 %call3.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body.i
  %70 = load i64, ptr %Other.i
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp5.ascast.i, i64 %70) 
  %call6.i = call spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %this1.i78, ptr byval(%"range") %agg.tmp5.i) 
  %71 = load i64, ptr addrspace(4) %call6.i
  %72 = load i64, ptr %Visible.i
  %add.i = add i64 %72, %71
  store i64 %add.i, ptr %Visible.i
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.body.i
  %73 = load i64, ptr %Other.i
  %inc.i = add i64 %73, 1
  store i64 %inc.i, ptr %Other.i
  br label %for.cond.i
}

define internal spir_func void @Foo40(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) align 1 %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @Init6(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}

define internal spir_func void @Foo41(ptr addrspace(4) dead_on_unwind noalias writable sret(%"nd_item") align 1 %agg.result, ptr addrspace(4) align 1 %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  ret void
}




define internal spir_func void @Foo42(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) %lhs, ptr addrspace(4) align 4 %rhs) {
entry:
  %lhs.addr = alloca ptr addrspace(4)
  %rhs.addr = alloca ptr addrspace(4)
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  store ptr addrspace(4) %lhs, ptr %lhs.addr
  store ptr addrspace(4) %rhs, ptr %rhs.addr
  call spir_func void @Foo11(ptr addrspace(4) %agg.result) 
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %1 = load ptr addrspace(4), ptr %lhs.addr
  %common_array1 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array1, i64 0, i64 %idxprom
  %3 = load i64, ptr addrspace(4) %arrayidx
  %4 = load ptr addrspace(4), ptr %rhs.addr
  %5 = load i32, ptr addrspace(4) %4, align 4
  %conv = sext i32 %5 to i64
  %cmp1 = icmp ult i64 %3, %conv
  %conv2 = zext i1 %cmp1 to i64
  %common_array32 = bitcast ptr addrspace(4) %agg.result to ptr addrspace(4)
  %6 = load i32, ptr %i, align 4
  %idxprom4 = sext i32 %6 to i64
  %arrayidx5 = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array32, i64 0, i64 %idxprom4
  store i64 %conv2, ptr addrspace(4) %arrayidx5
  %7 = load i32, ptr %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
}

declare void @llvm.assume(i1) 


define internal spir_func void @Foo43(ptr addrspace(4) dead_on_unwind noalias writable sret(%"tangle_group") %agg.result, ptr byval(%"nd_item") align 1 %group) {
entry:
  %mask = alloca %"ss_sub_group_mask"
  %agg.tmp = alloca %"nd_item", align 1
  %agg.tmp1 = alloca %"ss_sub_group_mask"
  %cleanup.dest.slot = alloca i32, align 4
  %mask.ascast = addrspacecast ptr %mask to ptr addrspace(4)
  %group.ascast = addrspacecast ptr %group to ptr addrspace(4)
  call spir_func void @Foo44(ptr addrspace(4) dead_on_unwind writable sret(%"ss_sub_group_mask") %mask.ascast, ptr byval(%"nd_item") align 1 %agg.tmp, i1 zeroext true) 
  call spir_func void @Foo45(ptr addrspace(4) %agg.result, ptr byval(%"ss_sub_group_mask") %agg.tmp1) 
  ret void
}


define internal spir_func void @Foo46(ptr addrspace(4) %this, i64 %dim0) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %dim0.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %dim0, ptr %dim0.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i64, ptr %dim0.addr
  call spir_func void @Foo60(ptr addrspace(4) %this1, i64 %0) 
  ret void
}


define internal spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %this, ptr byval(%"range") %Index) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %LinearIndex = alloca i64
  %agg.tmp = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %call = call spir_func i64 @Foo93(ptr addrspace(4) %this1, ptr byval(%"range") %agg.tmp) 
  store i64 %call, ptr %LinearIndex
  %call2 = call spir_func ptr addrspace(1) @Foo94(ptr addrspace(4) %this1) 
  %0 = load i64, ptr %LinearIndex
  %arrayidx = getelementptr inbounds nuw i64, ptr addrspace(1) %call2, i64 %0
  %arrayidx.ascast = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  ret ptr addrspace(4) %arrayidx.ascast
}


define internal spir_func void @Foo75(ptr byval(%"tangle_group") %G, i32 %FenceScope) {
entry:
  %FenceScope.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  store i32 %FenceScope, ptr %FenceScope.addr, align 4
  %0 = load i32, ptr %FenceScope.addr, align 4
  call spir_func void @Foo95(ptr byval(%"tangle_group") %agg.tmp, i32 %0, i32 5) 
  ret void
}


define internal spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %this, ptr byval(%"range") %Index) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %LinearIndex = alloca i64
  %agg.tmp = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %call = call spir_func i64 @Foo77(ptr addrspace(4) %this1, ptr byval(%"range") %agg.tmp) 
  store i64 %call, ptr %LinearIndex
  %call2 = call spir_func ptr addrspace(1) @Foo78(ptr addrspace(4) %this1) 
  %0 = load i64, ptr %LinearIndex
  %arrayidx = getelementptr inbounds nuw i8, ptr addrspace(1) %call2, i64 %0
  %arrayidx.ascast = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  ret ptr addrspace(4) %arrayidx.ascast
}


define internal spir_func i32 @Foo76(ptr addrspace(4) align 1 %this) {
entry:
  %retval = alloca i32, align 4
  %this.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @Foo96(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) align 1 %this1) 
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %ref.tmp.ascast, i32 0) 
  %0 = load i64, ptr addrspace(4) %call
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}


define internal spir_func i32 @Foo90(ptr addrspace(4) %this) {
entry:
  %retval = alloca i32, align 4
  %this.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) %this1) 
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %ref.tmp.ascast, i32 0) 
  %0 = load i64, ptr addrspace(4) %call
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}


define internal spir_func i32 @Foo91(ptr byval(%"tangle_group") %g, i32 %x, i32 %linear_local_id) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %linear_local_id.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"range"
  %agg.tmp2 = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %agg.tmp1.ascast = addrspacecast ptr %agg.tmp1 to ptr addrspace(4)
  %agg.tmp2.ascast = addrspacecast ptr %agg.tmp2 to ptr addrspace(4)
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i32 %linear_local_id, ptr %linear_local_id.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  call spir_func void @Foo97(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp2.ascast, ptr addrspace(4) %g.ascast) 
  %1 = load i32, ptr %linear_local_id.addr, align 4
  %conv = zext i32 %1 to i64
  call spir_func void @Foo98(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp1.ascast, ptr byval(%"range") %agg.tmp2, i64 %conv) 
  %call = call spir_func i32 @Bar69(ptr byval(%"tangle_group") %agg.tmp, i32 %0, ptr byval(%"range") %agg.tmp1) 
  ret i32 %call
}


define internal spir_func zeroext i1 @Foo92(ptr byval(%"tangle_group") %g, i1 zeroext %pred) {
entry:
  %retval = alloca i1, align 1
  %pred.addr = alloca i8, align 1
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %storedv = zext i1 %pred to i8
  store i8 %storedv, ptr %pred.addr, align 1
  %0 = load i8, ptr %pred.addr, align 1  
  %loadedv = trunc i8 %0 to i1
  %call = call spir_func zeroext i1 @Bar10(ptr byval(%"tangle_group") %agg.tmp, i1 zeroext %loadedv) 
  ret i1 %call
}


define internal spir_func zeroext i1 @Foo67(ptr byval(%"tangle_group") %g, i1 zeroext %pred) {
entry:
  %retval = alloca i1, align 1
  %pred.addr = alloca i8, align 1
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %storedv = zext i1 %pred to i8
  store i8 %storedv, ptr %pred.addr, align 1
  %0 = load i8, ptr %pred.addr, align 1  
  %loadedv = trunc i8 %0 to i1
  %call = call spir_func zeroext i1 @Foo66(ptr byval(%"tangle_group") %agg.tmp, i1 zeroext %loadedv) 
  ret i1 %call
}


define internal spir_func zeroext i1 @Foo65(ptr byval(%"tangle_group") %g, i1 zeroext %pred) {
entry:
  %retval = alloca i1, align 1
  %pred.addr = alloca i8, align 1
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %storedv = zext i1 %pred to i8
  store i8 %storedv, ptr %pred.addr, align 1
  %0 = load i8, ptr %pred.addr, align 1  
  %loadedv = trunc i8 %0 to i1
  %lnot = xor i1 %loadedv, true
  %call = call spir_func zeroext i1 @Foo66(ptr byval(%"tangle_group") %agg.tmp, i1 zeroext %lnot) 
  ret i1 %call
}


define internal spir_func i32 @Foo64(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"nd_item") align 1 %binary_op) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"nd_item", align 1
  %agg.tmp2 = alloca %"nd_item", align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %binary_op.ascast = addrspacecast ptr %binary_op to ptr addrspace(4)
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar11(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"nd_item") align 1 %agg.tmp1, i32 %0, ptr byval(%"nd_item") align 1 %agg.tmp2) 
  ret i32 %call
}


define internal spir_func i32 @Foo63(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"nd_item") align 1 %binary_op) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %res = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"nd_item", align 1
  %agg.tmp2 = alloca %"nd_item", align 1
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %binary_op.ascast = addrspacecast ptr %binary_op to ptr addrspace(4)
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar12(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"nd_item") align 1 %agg.tmp1, i32 %0, ptr byval(%"nd_item") align 1 %agg.tmp2) 
  store i32 %call, ptr %res, align 4
  %1 = load i32, ptr %res, align 4
  ret i32 %1
}


define internal spir_func i32 @Foo62(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"nd_item") align 1 %binary_op) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"nd_item", align 1
  %agg.tmp2 = alloca %"nd_item", align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %binary_op.ascast = addrspacecast ptr %binary_op to ptr addrspace(4)
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Foo61(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"nd_item") align 1 %agg.tmp1, i32 %0, ptr byval(%"nd_item") align 1 %agg.tmp2) 
  ret i32 %call
}


define internal spir_func i32 @Foo73(ptr byval(%"tangle_group") %g, i32 %x, i32 %delta) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %delta.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i32 %delta, ptr %delta.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %1 = load i32, ptr %delta.addr, align 4
  %call = call spir_func i32 @Foo72(ptr byval(%"tangle_group") %agg.tmp, i32 %0, i32 %1) 
  ret i32 %call
}


define internal spir_func zeroext i1 @Foo71(ptr addrspace(4) align 1 %this, i64 %Other) {
entry:
  %retval = alloca i1, align 1
  %this.addr = alloca ptr addrspace(4)
  %Other.addr = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %Other, ptr %Other.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i64, ptr %Other.addr
  %cmp = icmp ult i64 %0, 4
  ret i1 %cmp
}


define internal spir_func i32 @Foo53(ptr byval(%"tangle_group") %g, i32 %x, i32 %delta) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %delta.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i32 %delta, ptr %delta.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %1 = load i32, ptr %delta.addr, align 4
  %call = call spir_func i32 @Foo52(ptr byval(%"tangle_group") %agg.tmp, i32 %0, i32 %1) 
  ret i32 %call
}


define internal spir_func void @Foo51(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"ss_sub_group_mask"
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %Mask1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %call = call spir_func i32 @Foo47(ptr byval(%"ss_sub_group_mask") %agg.tmp) 
  %conv = zext i32 %call to i64
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %conv) 
  ret void
}


define internal spir_func void @Foo55(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) %lhs, ptr addrspace(4) align 4 %rhs) {
entry:
  %lhs.addr = alloca ptr addrspace(4)
  %rhs.addr = alloca ptr addrspace(4)
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  store ptr addrspace(4) %lhs, ptr %lhs.addr
  store ptr addrspace(4) %rhs, ptr %rhs.addr
  call spir_func void @Foo11(ptr addrspace(4) %agg.result) 
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %1 = load ptr addrspace(4), ptr %lhs.addr
  %common_array2 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array2, i64 0, i64 %idxprom
  %3 = load i64, ptr addrspace(4) %arrayidx
  %4 = load ptr addrspace(4), ptr %rhs.addr
  %5 = load i32, ptr addrspace(4) %4, align 4
  %conv = sext i32 %5 to i64
  %add = add i64 %3, %conv
  %common_array13 = bitcast ptr addrspace(4) %agg.result to ptr addrspace(4)
  %6 = load i32, ptr %i, align 4
  %idxprom2 = sext i32 %6 to i64
  %arrayidx3 = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array13, i64 0, i64 %idxprom2
  store i64 %add, ptr addrspace(4) %arrayidx3
  %7 = load i32, ptr %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
}


define internal spir_func void @Foo56(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) %lhs, ptr addrspace(4) %rhs) {
entry:
  %lhs.addr = alloca ptr addrspace(4)
  %rhs.addr = alloca ptr addrspace(4)
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  store ptr addrspace(4) %lhs, ptr %lhs.addr
  store ptr addrspace(4) %rhs, ptr %rhs.addr
  call spir_func void @Foo11(ptr addrspace(4) %agg.result) 
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %1 = load ptr addrspace(4), ptr %lhs.addr
  %common_array2 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array2, i64 0, i64 %idxprom
  %3 = load i64, ptr addrspace(4) %arrayidx
  %4 = load ptr addrspace(4), ptr %rhs.addr
  %5 = load i64, ptr addrspace(4) %4
  %rem = urem i64 %3, %5
  %common_array13 = bitcast ptr addrspace(4) %agg.result to ptr addrspace(4)
  %6 = load i32, ptr %i, align 4
  %idxprom2 = sext i32 %6 to i64
  %arrayidx3 = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array13, i64 0, i64 %idxprom2
  store i64 %rem, ptr addrspace(4) %arrayidx3
  %7 = load i32, ptr %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
}


define internal spir_func i32 @Foo57(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"range") %local_id) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Foo59(ptr byval(%"tangle_group") %agg.tmp, i32 %0, ptr byval(%"range") %agg.tmp1) 
  ret i32 %call
}


define internal spir_func i32 @Foo58(ptr byval(%"tangle_group") %g, i32 %x, i32 %mask) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %mask.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %agg.tmp1.ascast = addrspacecast ptr %agg.tmp1 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i32 %mask, ptr %mask.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %1 = load i32, ptr %mask.addr, align 4
  %conv = zext i32 %1 to i64
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp1.ascast, i64 %conv) 
  %call = call spir_func i32 @Bar13(ptr byval(%"tangle_group") %agg.tmp, i32 %0, ptr byval(%"range") %agg.tmp1) 
  ret i32 %call
}


define internal spir_func void @Foo68(ptr addrspace(4) %this, i64 %WI, ptr byval(%"tangle_group") %Tangle, i64 %TangleLeader, i64 %TangleSize, ptr byval(%"nd_item") align 1 %IsMember) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %WI.addr = alloca i64
  %TangleLeader.addr = alloca i64
  %TangleSize.addr = alloca i64
  %agg.tmp = alloca %"range"
  %agg.tmp2 = alloca %"tangle_group"
  %Visible = alloca i64
  %Other = alloca i64
  %cleanup.dest.slot = alloca i32, align 4
  %agg.tmp5 = alloca %"range"
  %agg.tmp8 = alloca %"range"
  %OriginalLID = alloca i32, align 4
  %LID = alloca i32, align 4
  %BroadcastResult = alloca i32, align 4
  %agg.tmp12 = alloca %"tangle_group"
  %agg.tmp15 = alloca %"range"
  %AnyResult = alloca i8, align 1
  %agg.tmp18 = alloca %"tangle_group"
  %agg.tmp24 = alloca %"range"
  %AllResult = alloca i8, align 1
  %agg.tmp27 = alloca %"tangle_group"
  %agg.tmp35 = alloca %"range"
  %NoneResult = alloca i8, align 1
  %agg.tmp38 = alloca %"tangle_group"
  %agg.tmp46 = alloca %"range"
  %ReduceResult = alloca i32, align 4
  %agg.tmp49 = alloca %"tangle_group"
  %agg.tmp50 = alloca %"nd_item", align 1
  %agg.tmp54 = alloca %"range"
  %ExScanResult = alloca i32, align 4
  %agg.tmp57 = alloca %"tangle_group"
  %agg.tmp58 = alloca %"nd_item", align 1
  %agg.tmp61 = alloca %"range"
  %IncScanResult = alloca i32, align 4
  %agg.tmp64 = alloca %"tangle_group"
  %agg.tmp65 = alloca %"nd_item", align 1
  %agg.tmp69 = alloca %"range"
  %ShiftLeftResult = alloca i32, align 4
  %agg.tmp72 = alloca %"tangle_group"
  %agg.tmp79 = alloca %"range"
  %ShiftRightResult = alloca i32, align 4
  %agg.tmp82 = alloca %"tangle_group"
  %agg.tmp88 = alloca %"range"
  %SelectResult = alloca i32, align 4
  %agg.tmp91 = alloca %"tangle_group"
  %agg.tmp92 = alloca %"range"
  %ref.tmp = alloca %"range"
  %ref.tmp93 = alloca %"range"
  %ref.tmp94 = alloca i32, align 4
  %agg.tmp100 = alloca %"range"
  %PermuteXorResult = alloca i32, align 4
  %agg.tmp103 = alloca %"tangle_group"
  %agg.tmp106 = alloca %"range"
  %TangleSize.addr.ascast = addrspacecast ptr %TangleSize.addr to ptr addrspace(4)
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %agg.tmp5.ascast = addrspacecast ptr %agg.tmp5 to ptr addrspace(4)
  %agg.tmp8.ascast = addrspacecast ptr %agg.tmp8 to ptr addrspace(4)
  %agg.tmp15.ascast = addrspacecast ptr %agg.tmp15 to ptr addrspace(4)
  %agg.tmp24.ascast = addrspacecast ptr %agg.tmp24 to ptr addrspace(4)
  %agg.tmp35.ascast = addrspacecast ptr %agg.tmp35 to ptr addrspace(4)
  %agg.tmp46.ascast = addrspacecast ptr %agg.tmp46 to ptr addrspace(4)
  %agg.tmp54.ascast = addrspacecast ptr %agg.tmp54 to ptr addrspace(4)
  %agg.tmp61.ascast = addrspacecast ptr %agg.tmp61 to ptr addrspace(4)
  %agg.tmp69.ascast = addrspacecast ptr %agg.tmp69 to ptr addrspace(4)
  %agg.tmp79.ascast = addrspacecast ptr %agg.tmp79 to ptr addrspace(4)
  %agg.tmp88.ascast = addrspacecast ptr %agg.tmp88 to ptr addrspace(4)
  %agg.tmp92.ascast = addrspacecast ptr %agg.tmp92 to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %ref.tmp93.ascast = addrspacecast ptr %ref.tmp93 to ptr addrspace(4)
  %ref.tmp94.ascast = addrspacecast ptr %ref.tmp94 to ptr addrspace(4)
  %agg.tmp100.ascast = addrspacecast ptr %agg.tmp100 to ptr addrspace(4)
  %agg.tmp106.ascast = addrspacecast ptr %agg.tmp106 to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %WI, ptr %WI.addr
  %Tangle.ascast = addrspacecast ptr %Tangle to ptr addrspace(4)
  store i64 %TangleLeader, ptr %TangleLeader.addr
  store i64 %TangleSize, ptr %TangleSize.addr
  %IsMember.ascast = addrspacecast ptr %IsMember to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  %TmpAcc1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp.ascast, i64 %0) 
  %call = call spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %TmpAcc1, ptr byval(%"range") %agg.tmp) 
  store i64 1, ptr addrspace(4) %call
  call spir_func void @Foo75(ptr byval(%"tangle_group") %agg.tmp2, i32 1) 
  store i64 0, ptr %Visible
  store i64 0, ptr %Other
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %1 = load i64, ptr %Other
  %cmp = icmp ult i64 %1, 32
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %2 = load i64, ptr %Visible
  %3 = load i64, ptr %TangleSize.addr
  %cmp7 = icmp eq i64 %2, %3
  %BarrierAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 1
  %4 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp8.ascast, i64 %4) 
  %call9 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %BarrierAcc, ptr byval(%"range") %agg.tmp8) 
  %storedv = zext i1 %cmp7 to i8
  store i8 %storedv, ptr addrspace(4) %call9, align 1
  %5 = getelementptr inbounds i8, ptr addrspace(4) %this1, i64 64
  %call10 = call spir_func i32 @Foo76(ptr addrspace(4) align 1 %5) 
  store i32 %call10, ptr %OriginalLID, align 4
  %call11 = call spir_func i32 @Foo90(ptr addrspace(4) %Tangle.ascast) 
  store i32 %call11, ptr %LID, align 4
  %6 = load i32, ptr %OriginalLID, align 4
  %call13 = call spir_func i32 @Foo91(ptr byval(%"tangle_group") %agg.tmp12, i32 %6, i32 0) 
  store i32 %call13, ptr %BroadcastResult, align 4
  %7 = load i32, ptr %BroadcastResult, align 4
  %conv = zext i32 %7 to i64
  %8 = load i64, ptr %TangleLeader.addr
  %cmp14 = icmp eq i64 %conv, %8
  %BroadcastAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 3
  %9 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp15.ascast, i64 %9) 
  %call16 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %BroadcastAcc, ptr byval(%"range") %agg.tmp15) 
  %storedv17 = zext i1 %cmp14 to i8
  store i8 %storedv17, ptr addrspace(4) %call16, align 1
  %10 = load i32, ptr %LID, align 4
  %cmp19 = icmp eq i32 %10, 0
  %call20 = call spir_func zeroext i1 @Foo92(ptr byval(%"tangle_group") %agg.tmp18, i1 zeroext %cmp19) 
  %storedv21 = zext i1 %call20 to i8
  store i8 %storedv21, ptr %AnyResult, align 1
  %11 = load i8, ptr %AnyResult, align 1  
  %loadedv = trunc i8 %11 to i1
  %conv22 = zext i1 %loadedv to i32
  %cmp23 = icmp eq i32 %conv22, 1
  %AnyAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 4
  %12 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp24.ascast, i64 %12) 
  %call25 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %AnyAcc, ptr byval(%"range") %agg.tmp24) 
  %storedv26 = zext i1 %cmp23 to i8
  store i8 %storedv26, ptr addrspace(4) %call25, align 1
  %13 = load i32, ptr %LID, align 4
  %conv28 = zext i32 %13 to i64
  %14 = load i64, ptr %TangleSize.addr
  %cmp29 = icmp ult i64 %conv28, %14
  %call30 = call spir_func zeroext i1 @Foo67(ptr byval(%"tangle_group") %agg.tmp27, i1 zeroext %cmp29) 
  %storedv31 = zext i1 %call30 to i8
  store i8 %storedv31, ptr %AllResult, align 1
  %15 = load i8, ptr %AllResult, align 1  
  %loadedv32 = trunc i8 %15 to i1
  %conv33 = zext i1 %loadedv32 to i32
  %cmp34 = icmp eq i32 %conv33, 1
  %AllAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 5
  %16 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp35.ascast, i64 %16) 
  %call36 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %AllAcc, ptr byval(%"range") %agg.tmp35) 
  %storedv37 = zext i1 %cmp34 to i8
  store i8 %storedv37, ptr addrspace(4) %call36, align 1
  %17 = load i32, ptr %LID, align 4
  %conv39 = zext i32 %17 to i64
  %18 = load i64, ptr %TangleSize.addr
  %cmp40 = icmp uge i64 %conv39, %18
  %call41 = call spir_func zeroext i1 @Foo65(ptr byval(%"tangle_group") %agg.tmp38, i1 zeroext %cmp40) 
  %storedv42 = zext i1 %call41 to i8
  store i8 %storedv42, ptr %NoneResult, align 1
  %19 = load i8, ptr %NoneResult, align 1  
  %loadedv43 = trunc i8 %19 to i1
  %conv44 = zext i1 %loadedv43 to i32
  %cmp45 = icmp eq i32 %conv44, 1
  %NoneAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 6
  %20 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp46.ascast, i64 %20) 
  %call47 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %NoneAcc, ptr byval(%"range") %agg.tmp46) 
  %storedv48 = zext i1 %cmp45 to i8
  store i8 %storedv48, ptr addrspace(4) %call47, align 1
  %call51 = call spir_func i32 @Foo64(ptr byval(%"tangle_group") %agg.tmp49, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp50) 
  store i32 %call51, ptr %ReduceResult, align 4
  %21 = load i32, ptr %ReduceResult, align 4
  %conv52 = zext i32 %21 to i64
  %22 = load i64, ptr %TangleSize.addr
  %cmp53 = icmp eq i64 %conv52, %22
  %ReduceAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 7
  %23 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp54.ascast, i64 %23) 
  %call55 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ReduceAcc, ptr byval(%"range") %agg.tmp54) 
  %storedv56 = zext i1 %cmp53 to i8
  store i8 %storedv56, ptr addrspace(4) %call55, align 1
  %call59 = call spir_func i32 @Foo63(ptr byval(%"tangle_group") %agg.tmp57, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp58) 
  store i32 %call59, ptr %ExScanResult, align 4
  %24 = load i32, ptr %ExScanResult, align 4
  %25 = load i32, ptr %LID, align 4
  %cmp60 = icmp eq i32 %24, %25
  %ExScanAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 8
  %26 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp61.ascast, i64 %26) 
  %call62 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ExScanAcc, ptr byval(%"range") %agg.tmp61) 
  %storedv63 = zext i1 %cmp60 to i8
  store i8 %storedv63, ptr addrspace(4) %call62, align 1
  %call66 = call spir_func i32 @Foo62(ptr byval(%"tangle_group") %agg.tmp64, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp65) 
  store i32 %call66, ptr %IncScanResult, align 4
  %27 = load i32, ptr %IncScanResult, align 4
  %28 = load i32, ptr %LID, align 4
  %add67 = add i32 %28, 1
  %cmp68 = icmp eq i32 %27, %add67
  %IncScanAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 9
  %29 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp69.ascast, i64 %29) 
  %call70 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %IncScanAcc, ptr byval(%"range") %agg.tmp69) 
  %storedv71 = zext i1 %cmp68 to i8
  store i8 %storedv71, ptr addrspace(4) %call70, align 1
  %30 = load i32, ptr %LID, align 4
  %call73 = call spir_func i32 @Foo73(ptr byval(%"tangle_group") %agg.tmp72, i32 %30, i32 2) 
  store i32 %call73, ptr %ShiftLeftResult, align 4
  %31 = load i32, ptr %LID, align 4
  %add74 = add i32 %31, 2
  %conv75 = zext i32 %add74 to i64
  %32 = load i64, ptr %TangleSize.addr
  %cmp76 = icmp uge i64 %conv75, %32
  br i1 %cmp76, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %for.cond.cleanup
  %33 = load i32, ptr %ShiftLeftResult, align 4
  %34 = load i32, ptr %LID, align 4
  %add77 = add i32 %34, 2
  %cmp78 = icmp eq i32 %33, %add77
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %for.cond.cleanup
  %35 = phi i1 [ true, %for.cond.cleanup ], [ %cmp78, %lor.rhs ]
  %ShiftLeftAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 10
  %36 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp79.ascast, i64 %36) 
  %call80 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ShiftLeftAcc, ptr byval(%"range") %agg.tmp79) 
  %storedv81 = zext i1 %35 to i8
  store i8 %storedv81, ptr addrspace(4) %call80, align 1
  %37 = load i32, ptr %LID, align 4
  %call83 = call spir_func i32 @Foo53(ptr byval(%"tangle_group") %agg.tmp82, i32 %37, i32 2) 
  store i32 %call83, ptr %ShiftRightResult, align 4
  %38 = load i32, ptr %LID, align 4
  %cmp84 = icmp ult i32 %38, 2
  br i1 %cmp84, label %lor.end87, label %lor.rhs85

lor.rhs85:                                        ; preds = %lor.end
  %39 = load i32, ptr %ShiftRightResult, align 4
  %40 = load i32, ptr %LID, align 4
  %sub = sub i32 %40, 2
  %cmp86 = icmp eq i32 %39, %sub
  br label %lor.end87

lor.end87:                                        ; preds = %lor.rhs85, %lor.end
  %41 = phi i1 [ true, %lor.end ], [ %cmp86, %lor.rhs85 ]
  %ShiftRightAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 11
  %42 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp88.ascast, i64 %42) 
  %call89 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ShiftRightAcc, ptr byval(%"range") %agg.tmp88) 
  %storedv90 = zext i1 %41 to i8
  store i8 %storedv90, ptr addrspace(4) %call89, align 1
  %43 = load i32, ptr %LID, align 4
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp93.ascast, ptr addrspace(4) %Tangle.ascast) 
  store i32 2, ptr %ref.tmp94, align 4
  call spir_func void @Foo55(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) %ref.tmp93.ascast, ptr addrspace(4) align 4 %ref.tmp94.ascast) 
  call spir_func void @Foo56(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp92.ascast, ptr addrspace(4) %ref.tmp.ascast, ptr addrspace(4) %TangleSize.addr.ascast) 
  %call95 = call spir_func i32 @Foo57(ptr byval(%"tangle_group") %agg.tmp91, i32 %43, ptr byval(%"range") %agg.tmp92) 
  store i32 %call95, ptr %SelectResult, align 4
  %44 = load i32, ptr %SelectResult, align 4
  %conv96 = zext i32 %44 to i64
  %45 = load i32, ptr %LID, align 4
  %add97 = add i32 %45, 2
  %conv98 = zext i32 %add97 to i64
  %46 = load i64, ptr %TangleSize.addr
  %rem = urem i64 %conv98, %46
  %cmp99 = icmp eq i64 %conv96, %rem
  %SelectAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 12
  %47 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp100.ascast, i64 %47) 
  %call101 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %SelectAcc, ptr byval(%"range") %agg.tmp100) 
  %storedv102 = zext i1 %cmp99 to i8
  store i8 %storedv102, ptr addrspace(4) %call101, align 1
  %48 = load i32, ptr %LID, align 4
  %call104 = call spir_func i32 @Foo58(ptr byval(%"tangle_group") %agg.tmp103, i32 %48, i32 2) 
  store i32 %call104, ptr %PermuteXorResult, align 4
  %49 = load i32, ptr %PermuteXorResult, align 4
  %50 = load i32, ptr %LID, align 4
  %xor = xor i32 %50, 2
  %cmp105 = icmp eq i32 %49, %xor
  %PermuteXorAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 13
  %51 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp106.ascast, i64 %51) 
  %call107 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %PermuteXorAcc, ptr byval(%"range") %agg.tmp106) 
  %storedv108 = zext i1 %cmp105 to i8
  store i8 %storedv108, ptr addrspace(4) %call107, align 1
  ret void

for.body:                                         ; preds = %for.cond
  %52 = load i64, ptr %Other
  %call3 = call spir_func zeroext i1 @Foo74(ptr addrspace(4) align 1 %IsMember.ascast, i64 %52) 
  br i1 %call3, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %TmpAcc42 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %53 = load i64, ptr %Other
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp5.ascast, i64 %53) 
  %call6 = call spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %TmpAcc42, ptr byval(%"range") %agg.tmp5) 
  %54 = load i64, ptr addrspace(4) %call6
  %55 = load i64, ptr %Visible
  %add = add i64 %55, %54
  store i64 %add, ptr %Visible
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %56 = load i64, ptr %Other
  %inc = add i64 %56, 1
  store i64 %inc, ptr %Other
  br label %for.cond
}


define internal spir_func void @Foo69(ptr addrspace(4) %this, i64 %WI, ptr byval(%"tangle_group") %Tangle, i64 %TangleLeader, i64 %TangleSize, ptr byval(%"nd_item") align 1 %IsMember) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %WI.addr = alloca i64
  %TangleLeader.addr = alloca i64
  %TangleSize.addr = alloca i64
  %agg.tmp = alloca %"range"
  %agg.tmp2 = alloca %"tangle_group"
  %Visible = alloca i64
  %Other = alloca i64
  %cleanup.dest.slot = alloca i32, align 4
  %agg.tmp5 = alloca %"range"
  %agg.tmp8 = alloca %"range"
  %OriginalLID = alloca i32, align 4
  %LID = alloca i32, align 4
  %BroadcastResult = alloca i32, align 4
  %agg.tmp12 = alloca %"tangle_group"
  %agg.tmp15 = alloca %"range"
  %AnyResult = alloca i8, align 1
  %agg.tmp18 = alloca %"tangle_group"
  %agg.tmp24 = alloca %"range"
  %AllResult = alloca i8, align 1
  %agg.tmp27 = alloca %"tangle_group"
  %agg.tmp35 = alloca %"range"
  %NoneResult = alloca i8, align 1
  %agg.tmp38 = alloca %"tangle_group"
  %agg.tmp46 = alloca %"range"
  %ReduceResult = alloca i32, align 4
  %agg.tmp49 = alloca %"tangle_group"
  %agg.tmp50 = alloca %"nd_item", align 1
  %agg.tmp54 = alloca %"range"
  %ExScanResult = alloca i32, align 4
  %agg.tmp57 = alloca %"tangle_group"
  %agg.tmp58 = alloca %"nd_item", align 1
  %agg.tmp61 = alloca %"range"
  %IncScanResult = alloca i32, align 4
  %agg.tmp64 = alloca %"tangle_group"
  %agg.tmp65 = alloca %"nd_item", align 1
  %agg.tmp69 = alloca %"range"
  %ShiftLeftResult = alloca i32, align 4
  %agg.tmp72 = alloca %"tangle_group"
  %agg.tmp79 = alloca %"range"
  %ShiftRightResult = alloca i32, align 4
  %agg.tmp82 = alloca %"tangle_group"
  %agg.tmp88 = alloca %"range"
  %SelectResult = alloca i32, align 4
  %agg.tmp91 = alloca %"tangle_group"
  %agg.tmp92 = alloca %"range"
  %ref.tmp = alloca %"range"
  %ref.tmp93 = alloca %"range"
  %ref.tmp94 = alloca i32, align 4
  %agg.tmp100 = alloca %"range"
  %PermuteXorResult = alloca i32, align 4
  %agg.tmp103 = alloca %"tangle_group"
  %agg.tmp106 = alloca %"range"
  %TangleSize.addr.ascast = addrspacecast ptr %TangleSize.addr to ptr addrspace(4)
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %agg.tmp5.ascast = addrspacecast ptr %agg.tmp5 to ptr addrspace(4)
  %agg.tmp8.ascast = addrspacecast ptr %agg.tmp8 to ptr addrspace(4)
  %agg.tmp15.ascast = addrspacecast ptr %agg.tmp15 to ptr addrspace(4)
  %agg.tmp24.ascast = addrspacecast ptr %agg.tmp24 to ptr addrspace(4)
  %agg.tmp35.ascast = addrspacecast ptr %agg.tmp35 to ptr addrspace(4)
  %agg.tmp46.ascast = addrspacecast ptr %agg.tmp46 to ptr addrspace(4)
  %agg.tmp54.ascast = addrspacecast ptr %agg.tmp54 to ptr addrspace(4)
  %agg.tmp61.ascast = addrspacecast ptr %agg.tmp61 to ptr addrspace(4)
  %agg.tmp69.ascast = addrspacecast ptr %agg.tmp69 to ptr addrspace(4)
  %agg.tmp79.ascast = addrspacecast ptr %agg.tmp79 to ptr addrspace(4)
  %agg.tmp88.ascast = addrspacecast ptr %agg.tmp88 to ptr addrspace(4)
  %agg.tmp92.ascast = addrspacecast ptr %agg.tmp92 to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %ref.tmp93.ascast = addrspacecast ptr %ref.tmp93 to ptr addrspace(4)
  %ref.tmp94.ascast = addrspacecast ptr %ref.tmp94 to ptr addrspace(4)
  %agg.tmp100.ascast = addrspacecast ptr %agg.tmp100 to ptr addrspace(4)
  %agg.tmp106.ascast = addrspacecast ptr %agg.tmp106 to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %WI, ptr %WI.addr
  %Tangle.ascast = addrspacecast ptr %Tangle to ptr addrspace(4)
  store i64 %TangleLeader, ptr %TangleLeader.addr
  store i64 %TangleSize, ptr %TangleSize.addr
  %IsMember.ascast = addrspacecast ptr %IsMember to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  %TmpAcc1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp.ascast, i64 %0) 
  %call = call spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %TmpAcc1, ptr byval(%"range") %agg.tmp) 
  store i64 1, ptr addrspace(4) %call
  call spir_func void @Foo75(ptr byval(%"tangle_group") %agg.tmp2, i32 1) 
  store i64 0, ptr %Visible
  store i64 0, ptr %Other
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %1 = load i64, ptr %Other
  %cmp = icmp ult i64 %1, 32
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %2 = load i64, ptr %Visible
  %3 = load i64, ptr %TangleSize.addr
  %cmp7 = icmp eq i64 %2, %3
  %BarrierAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 1
  %4 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp8.ascast, i64 %4) 
  %call9 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %BarrierAcc, ptr byval(%"range") %agg.tmp8) 
  %storedv = zext i1 %cmp7 to i8
  store i8 %storedv, ptr addrspace(4) %call9, align 1
  %5 = getelementptr inbounds i8, ptr addrspace(4) %this1, i64 64
  %call10 = call spir_func i32 @Foo76(ptr addrspace(4) align 1 %5) 
  store i32 %call10, ptr %OriginalLID, align 4
  %call11 = call spir_func i32 @Foo90(ptr addrspace(4) %Tangle.ascast) 
  store i32 %call11, ptr %LID, align 4
  %6 = load i32, ptr %OriginalLID, align 4
  %call13 = call spir_func i32 @Foo91(ptr byval(%"tangle_group") %agg.tmp12, i32 %6, i32 0) 
  store i32 %call13, ptr %BroadcastResult, align 4
  %7 = load i32, ptr %BroadcastResult, align 4
  %conv = zext i32 %7 to i64
  %8 = load i64, ptr %TangleLeader.addr
  %cmp14 = icmp eq i64 %conv, %8
  %BroadcastAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 3
  %9 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp15.ascast, i64 %9) 
  %call16 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %BroadcastAcc, ptr byval(%"range") %agg.tmp15) 
  %storedv17 = zext i1 %cmp14 to i8
  store i8 %storedv17, ptr addrspace(4) %call16, align 1
  %10 = load i32, ptr %LID, align 4
  %cmp19 = icmp eq i32 %10, 0
  %call20 = call spir_func zeroext i1 @Foo92(ptr byval(%"tangle_group") %agg.tmp18, i1 zeroext %cmp19) 
  %storedv21 = zext i1 %call20 to i8
  store i8 %storedv21, ptr %AnyResult, align 1
  %11 = load i8, ptr %AnyResult, align 1  
  %loadedv = trunc i8 %11 to i1
  %conv22 = zext i1 %loadedv to i32
  %cmp23 = icmp eq i32 %conv22, 1
  %AnyAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 4
  %12 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp24.ascast, i64 %12) 
  %call25 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %AnyAcc, ptr byval(%"range") %agg.tmp24) 
  %storedv26 = zext i1 %cmp23 to i8
  store i8 %storedv26, ptr addrspace(4) %call25, align 1
  %13 = load i32, ptr %LID, align 4
  %conv28 = zext i32 %13 to i64
  %14 = load i64, ptr %TangleSize.addr
  %cmp29 = icmp ult i64 %conv28, %14
  %call30 = call spir_func zeroext i1 @Foo67(ptr byval(%"tangle_group") %agg.tmp27, i1 zeroext %cmp29) 
  %storedv31 = zext i1 %call30 to i8
  store i8 %storedv31, ptr %AllResult, align 1
  %15 = load i8, ptr %AllResult, align 1  
  %loadedv32 = trunc i8 %15 to i1
  %conv33 = zext i1 %loadedv32 to i32
  %cmp34 = icmp eq i32 %conv33, 1
  %AllAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 5
  %16 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp35.ascast, i64 %16) 
  %call36 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %AllAcc, ptr byval(%"range") %agg.tmp35) 
  %storedv37 = zext i1 %cmp34 to i8
  store i8 %storedv37, ptr addrspace(4) %call36, align 1
  %17 = load i32, ptr %LID, align 4
  %conv39 = zext i32 %17 to i64
  %18 = load i64, ptr %TangleSize.addr
  %cmp40 = icmp uge i64 %conv39, %18
  %call41 = call spir_func zeroext i1 @Foo65(ptr byval(%"tangle_group") %agg.tmp38, i1 zeroext %cmp40) 
  %storedv42 = zext i1 %call41 to i8
  store i8 %storedv42, ptr %NoneResult, align 1
  %19 = load i8, ptr %NoneResult, align 1  
  %loadedv43 = trunc i8 %19 to i1
  %conv44 = zext i1 %loadedv43 to i32
  %cmp45 = icmp eq i32 %conv44, 1
  %NoneAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 6
  %20 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp46.ascast, i64 %20) 
  %call47 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %NoneAcc, ptr byval(%"range") %agg.tmp46) 
  %storedv48 = zext i1 %cmp45 to i8
  store i8 %storedv48, ptr addrspace(4) %call47, align 1
  %call51 = call spir_func i32 @Foo64(ptr byval(%"tangle_group") %agg.tmp49, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp50) 
  store i32 %call51, ptr %ReduceResult, align 4
  %21 = load i32, ptr %ReduceResult, align 4
  %conv52 = zext i32 %21 to i64
  %22 = load i64, ptr %TangleSize.addr
  %cmp53 = icmp eq i64 %conv52, %22
  %ReduceAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 7
  %23 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp54.ascast, i64 %23) 
  %call55 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ReduceAcc, ptr byval(%"range") %agg.tmp54) 
  %storedv56 = zext i1 %cmp53 to i8
  store i8 %storedv56, ptr addrspace(4) %call55, align 1
  %call59 = call spir_func i32 @Foo63(ptr byval(%"tangle_group") %agg.tmp57, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp58) 
  store i32 %call59, ptr %ExScanResult, align 4
  %24 = load i32, ptr %ExScanResult, align 4
  %25 = load i32, ptr %LID, align 4
  %cmp60 = icmp eq i32 %24, %25
  %ExScanAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 8
  %26 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp61.ascast, i64 %26) 
  %call62 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ExScanAcc, ptr byval(%"range") %agg.tmp61) 
  %storedv63 = zext i1 %cmp60 to i8
  store i8 %storedv63, ptr addrspace(4) %call62, align 1
  %call66 = call spir_func i32 @Foo62(ptr byval(%"tangle_group") %agg.tmp64, i32 1, ptr byval(%"nd_item") align 1 %agg.tmp65) 
  store i32 %call66, ptr %IncScanResult, align 4
  %27 = load i32, ptr %IncScanResult, align 4
  %28 = load i32, ptr %LID, align 4
  %add67 = add i32 %28, 1
  %cmp68 = icmp eq i32 %27, %add67
  %IncScanAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 9
  %29 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp69.ascast, i64 %29) 
  %call70 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %IncScanAcc, ptr byval(%"range") %agg.tmp69) 
  %storedv71 = zext i1 %cmp68 to i8
  store i8 %storedv71, ptr addrspace(4) %call70, align 1
  %30 = load i32, ptr %LID, align 4
  %call73 = call spir_func i32 @Foo73(ptr byval(%"tangle_group") %agg.tmp72, i32 %30, i32 2) 
  store i32 %call73, ptr %ShiftLeftResult, align 4
  %31 = load i32, ptr %LID, align 4
  %add74 = add i32 %31, 2
  %conv75 = zext i32 %add74 to i64
  %32 = load i64, ptr %TangleSize.addr
  %cmp76 = icmp uge i64 %conv75, %32
  br i1 %cmp76, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %for.cond.cleanup
  %33 = load i32, ptr %ShiftLeftResult, align 4
  %34 = load i32, ptr %LID, align 4
  %add77 = add i32 %34, 2
  %cmp78 = icmp eq i32 %33, %add77
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %for.cond.cleanup
  %35 = phi i1 [ true, %for.cond.cleanup ], [ %cmp78, %lor.rhs ]
  %ShiftLeftAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 10
  %36 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp79.ascast, i64 %36) 
  %call80 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ShiftLeftAcc, ptr byval(%"range") %agg.tmp79) 
  %storedv81 = zext i1 %35 to i8
  store i8 %storedv81, ptr addrspace(4) %call80, align 1
  %37 = load i32, ptr %LID, align 4
  %call83 = call spir_func i32 @Foo53(ptr byval(%"tangle_group") %agg.tmp82, i32 %37, i32 2) 
  store i32 %call83, ptr %ShiftRightResult, align 4
  %38 = load i32, ptr %LID, align 4
  %cmp84 = icmp ult i32 %38, 2
  br i1 %cmp84, label %lor.end87, label %lor.rhs85

lor.rhs85:                                        ; preds = %lor.end
  %39 = load i32, ptr %ShiftRightResult, align 4
  %40 = load i32, ptr %LID, align 4
  %sub = sub i32 %40, 2
  %cmp86 = icmp eq i32 %39, %sub
  br label %lor.end87

lor.end87:                                        ; preds = %lor.rhs85, %lor.end
  %41 = phi i1 [ true, %lor.end ], [ %cmp86, %lor.rhs85 ]
  %ShiftRightAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 11
  %42 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp88.ascast, i64 %42) 
  %call89 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %ShiftRightAcc, ptr byval(%"range") %agg.tmp88) 
  %storedv90 = zext i1 %41 to i8
  store i8 %storedv90, ptr addrspace(4) %call89, align 1
  %43 = load i32, ptr %LID, align 4
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp93.ascast, ptr addrspace(4) %Tangle.ascast) 
  store i32 2, ptr %ref.tmp94, align 4
  call spir_func void @Foo55(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) %ref.tmp93.ascast, ptr addrspace(4) align 4 %ref.tmp94.ascast) 
  call spir_func void @Foo56(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.tmp92.ascast, ptr addrspace(4) %ref.tmp.ascast, ptr addrspace(4) %TangleSize.addr.ascast) 
  %call95 = call spir_func i32 @Foo57(ptr byval(%"tangle_group") %agg.tmp91, i32 %43, ptr byval(%"range") %agg.tmp92) 
  store i32 %call95, ptr %SelectResult, align 4
  %44 = load i32, ptr %SelectResult, align 4
  %conv96 = zext i32 %44 to i64
  %45 = load i32, ptr %LID, align 4
  %add97 = add i32 %45, 2
  %conv98 = zext i32 %add97 to i64
  %46 = load i64, ptr %TangleSize.addr
  %rem = urem i64 %conv98, %46
  %cmp99 = icmp eq i64 %conv96, %rem
  %SelectAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 12
  %47 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp100.ascast, i64 %47) 
  %call101 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %SelectAcc, ptr byval(%"range") %agg.tmp100) 
  %storedv102 = zext i1 %cmp99 to i8
  store i8 %storedv102, ptr addrspace(4) %call101, align 1
  %48 = load i32, ptr %LID, align 4
  %call104 = call spir_func i32 @Foo58(ptr byval(%"tangle_group") %agg.tmp103, i32 %48, i32 2) 
  store i32 %call104, ptr %PermuteXorResult, align 4
  %49 = load i32, ptr %PermuteXorResult, align 4
  %50 = load i32, ptr %LID, align 4
  %xor = xor i32 %50, 2
  %cmp105 = icmp eq i32 %49, %xor
  %PermuteXorAcc = getelementptr inbounds nuw %class.anon.8, ptr addrspace(4) %this1, i32 0, i32 13
  %51 = load i64, ptr %WI.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp106.ascast, i64 %51) 
  %call107 = call spir_func align 1 ptr addrspace(4) @Foo54(ptr addrspace(4) %PermuteXorAcc, ptr byval(%"range") %agg.tmp106) 
  %storedv108 = zext i1 %cmp105 to i8
  store i8 %storedv108, ptr addrspace(4) %call107, align 1
  ret void

for.body:                                         ; preds = %for.cond
  %52 = load i64, ptr %Other
  %call3 = call spir_func zeroext i1 @Bar14(ptr addrspace(4) align 1 %IsMember.ascast, i64 %52) 
  br i1 %call3, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %TmpAcc42 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %53 = load i64, ptr %Other
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp5.ascast, i64 %53) 
  %call6 = call spir_func ptr addrspace(4) @Foo70(ptr addrspace(4) %TmpAcc42, ptr byval(%"range") %agg.tmp5) 
  %54 = load i64, ptr addrspace(4) %call6
  %55 = load i64, ptr %Visible
  %add = add i64 %55, %54
  store i64 %add, ptr %Visible
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %56 = load i64, ptr %Other
  %inc = add i64 %56, 1
  store i64 %inc, ptr %Other
  br label %for.cond
}


define internal spir_func zeroext i1 @Bar14(ptr addrspace(4) align 1 %this, i64 %Other) {
entry:
  %retval = alloca i1, align 1
  %this.addr = alloca ptr addrspace(4)
  %Other.addr = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %Other, ptr %Other.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i64, ptr %Other.addr
  %cmp = icmp uge i64 %0, 24
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
  %1 = load i64, ptr %Other.addr
  %cmp2 = icmp ult i64 %1, 32
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %2 = phi i1 [ false, %entry ], [ %cmp2, %land.rhs ]
  ret i1 %2
}


define internal spir_func zeroext i1 @Foo74(ptr addrspace(4) align 1 %this, i64 %Other) {
entry:
  %retval = alloca i1, align 1
  %this.addr = alloca ptr addrspace(4)
  %Other.addr = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %Other, ptr %Other.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i64, ptr %Other.addr
  %cmp = icmp uge i64 %0, 4
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
  %1 = load i64, ptr %Other.addr
  %cmp2 = icmp ult i64 %1, 24
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %2 = phi i1 [ false, %entry ], [ %cmp2, %land.rhs ]
  ret i1 %2
}


define internal spir_func i32 @Bar13(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"range") %mask) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %TargetLocalId = alloca %"range"
  %ref.tmp = alloca %"range"
  %TargetId = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %x.addr.ascast = addrspacecast ptr %x.addr to ptr addrspace(4)
  %TargetLocalId.ascast = addrspacecast ptr %TargetLocalId to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %mask.ascast = addrspacecast ptr %mask to ptr addrspace(4)
  %0 = addrspacecast ptr addrspace(1) @_ZSt6ignore to ptr addrspace(4)
  %call = call spir_func align 1 ptr addrspace(4) @Bar15(ptr addrspace(4) align 1 %0, ptr addrspace(4) %g.ascast) 
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) %g.ascast) 
  call spir_func void @Bar16(ptr addrspace(4) dead_on_unwind writable sret(%"range") %TargetLocalId.ascast, ptr addrspace(4) %ref.tmp.ascast, ptr addrspace(4) %mask.ascast) 
  %call2 = call spir_func i32 @Foo48(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"range") %agg.tmp1) 
  store i32 %call2, ptr %TargetId, align 4
  %call3 = call spir_func i32 @Foo49(ptr addrspace(4) align 4 %x.addr.ascast) 
  %1 = load i32, ptr %TargetId, align 4
  %call4 = call spir_func i32 @Foo50(i32 3, i32 %call3, i32 %1) 
  ret i32 %call4
}


define internal spir_func align 1 ptr addrspace(4) @Bar15(ptr addrspace(4) align 1 %this, ptr addrspace(4) %0) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %0, ptr %.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  ret ptr addrspace(4) %this1
}


define internal spir_func void @Bar16(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) %lhs, ptr addrspace(4) %rhs) {
entry:
  %lhs.addr = alloca ptr addrspace(4)
  %rhs.addr = alloca ptr addrspace(4)
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  store ptr addrspace(4) %lhs, ptr %lhs.addr
  store ptr addrspace(4) %rhs, ptr %rhs.addr
  call spir_func void @Foo11(ptr addrspace(4) %agg.result) 
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %1 = load ptr addrspace(4), ptr %lhs.addr
  %common_array2 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array2, i64 0, i64 %idxprom
  %3 = load i64, ptr addrspace(4) %arrayidx
  %4 = load ptr addrspace(4), ptr %rhs.addr
  %common_array13 = bitcast ptr addrspace(4) %4 to ptr addrspace(4)
  %5 = load i32, ptr %i, align 4
  %idxprom2 = sext i32 %5 to i64
  %arrayidx3 = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array13, i64 0, i64 %idxprom2
  %6 = load i64, ptr addrspace(4) %arrayidx3
  %xor = xor i64 %3, %6
  %common_array44 = bitcast ptr addrspace(4) %agg.result to ptr addrspace(4)
  %7 = load i32, ptr %i, align 4
  %idxprom5 = sext i32 %7 to i64
  %arrayidx6 = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array44, i64 0, i64 %idxprom5
  store i64 %xor, ptr addrspace(4) %arrayidx6
  %8 = load i32, ptr %i, align 4
  %inc = add nsw i32 %8, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
}


define internal spir_func i32 @Foo48(ptr byval(%"tangle_group") %g, ptr byval(%"range") %local_id) {
entry:
  %retval.i = alloca i64
  %this.addr.i = alloca ptr addrspace(4)
  %Result.i = alloca i64
  %retval = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %local_id.ascast = addrspacecast ptr %local_id to ptr addrspace(4)
  %retval.ascast.i = addrspacecast ptr %retval.i to ptr addrspace(4)
  store ptr addrspace(4) %local_id.ascast, ptr %this.addr.i
  %this1.i = load ptr addrspace(4), ptr %this.addr.i
  %0 = load i64, ptr addrspace(4) %this1.i
  store i64 %0, ptr %Result.i
  %1 = load i64, ptr %Result.i
  %conv = trunc i64 %1 to i32
  %call1 = call spir_func i32 @Bar17(ptr byval(%"tangle_group") %agg.tmp, i32 %conv) 
  ret i32 %call1
}


define internal spir_func i32 @Foo49(ptr addrspace(4) align 4 %x) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %x, ptr %x.addr
  %0 = load ptr addrspace(4), ptr %x.addr
  %1 = load i32, ptr addrspace(4) %0, align 4
  ret i32 %1
}

declare dso_local spir_func i32 @Foo50(i32, i32, i32) 


define internal spir_func i32 @Bar17(ptr byval(%"tangle_group") %Group, i32 %Id) {
entry:
  %retval = alloca i32, align 4
  %Id.addr = alloca i32, align 4
  %MemberMask = alloca %"vec.16", align 16
  %agg.tmp = alloca %"ss_sub_group_mask"
  %agg.tmp1 = alloca %"tangle_group"
  %Count = alloca i32, align 4
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %b = alloca i32, align 4
  %MemberMask.ascast = addrspacecast ptr %MemberMask to ptr addrspace(4)
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  store i32 %Id, ptr %Id.addr, align 4
  call spir_func void @Bar18(ptr addrspace(4) dead_on_unwind writable sret(%"ss_sub_group_mask") %agg.tmp.ascast, ptr byval(%"tangle_group") %agg.tmp1) 
  call spir_func void @Bar19(ptr addrspace(4) dead_on_unwind writable sret(%"vec.16") align 16 %MemberMask.ascast, ptr byval(%"ss_sub_group_mask") %agg.tmp) 
  store i32 0, ptr %Count, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, ptr %cleanup.dest.slot, align 4
  br label %cleanup12

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %b, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %if.end8, %for.body
  %1 = load i32, ptr %b, align 4
  %cmp3 = icmp slt i32 %1, 32
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %for.cond2
  store i32 5, ptr %cleanup.dest.slot, align 4
  br label %cleanup

for.body5:                                        ; preds = %for.cond2
  %2 = load i32, ptr %i, align 4
  %call = call spir_func align 4 ptr addrspace(4) @Bar20(ptr addrspace(4) align 16 %MemberMask.ascast, i32 %2) 
  %3 = load i32, ptr addrspace(4) %call, align 4
  %4 = load i32, ptr %b, align 4
  %shl = shl i32 1, %4
  %and = and i32 %3, %shl
  %tobool = icmp ne i32 %and, 0
  br i1 %tobool, label %if.then, label %if.end8

if.then:                                          ; preds = %for.body5
  %5 = load i32, ptr %Count, align 4
  %6 = load i32, ptr %Id.addr, align 4
  %cmp6 = icmp eq i32 %5, %6
  br i1 %cmp6, label %if.then7, label %if.end

if.end:                                           ; preds = %if.then
  %7 = load i32, ptr %Count, align 4
  %inc = add i32 %7, 1
  store i32 %inc, ptr %Count, align 4
  br label %if.end8

if.end8:                                          ; preds = %if.end, %for.body5
  %8 = load i32, ptr %b, align 4
  %inc9 = add nsw i32 %8, 1
  store i32 %inc9, ptr %b, align 4
  br label %for.cond2

if.then7:                                         ; preds = %if.then
  %9 = load i32, ptr %i, align 4
  %mul = mul nsw i32 %9, 32
  %10 = load i32, ptr %b, align 4
  %add = add nsw i32 %mul, %10
  store i32 %add, ptr %retval, align 4
  store i32 1, ptr %cleanup.dest.slot, align 4
  br label %cleanup

cleanup:                                          ; preds = %if.then7, %for.cond.cleanup4
  %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
  %cond = icmp eq i32 %cleanup.dest, 5
  br i1 %cond, label %for.end, label %cleanup12

for.end:                                          ; preds = %cleanup
  %11 = load i32, ptr %i, align 4
  %inc11 = add nsw i32 %11, 1
  store i32 %inc11, ptr %i, align 4
  br label %for.cond

cleanup12:                                        ; preds = %cleanup, %for.cond.cleanup
  %cleanup.dest13 = load i32, ptr %cleanup.dest.slot, align 4
  %cond1 = icmp eq i32 %cleanup.dest13, 2
  br i1 %cond1, label %for.end14, label %cleanup15

for.end14:                                        ; preds = %cleanup12
  %12 = load i32, ptr %Count, align 4
  store i32 %12, ptr %retval, align 4
  store i32 1, ptr %cleanup.dest.slot, align 4
  br label %cleanup15

cleanup15:                                        ; preds = %cleanup12, %for.end14
  %13 = load i32, ptr %retval, align 4
  ret i32 %13
}


define internal spir_func void @Bar18(ptr addrspace(4) dead_on_unwind noalias writable sret(%"ss_sub_group_mask") %agg.result, ptr byval(%"tangle_group") %Group) {
entry:
  %Mask1 = bitcast ptr %Group to ptr
  ret void
}


define internal spir_func void @Bar19(ptr addrspace(4) dead_on_unwind noalias writable sret(%"vec.16") align 16 %agg.result, ptr byval(%"ss_sub_group_mask") %Mask) {
entry:
  %TmpMArray = alloca %"struct.std::array.20", align 4
  %agg.tmp = alloca %"range"
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %cleanup.dest.slot2 = alloca i32, align 4
  %TmpMArray.ascast = addrspacecast ptr %TmpMArray to ptr addrspace(4)
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %Mask.ascast = addrspacecast ptr %Mask to ptr addrspace(4)
  call spir_func void @Bar50(ptr addrspace(4) align 4 %TmpMArray.ascast) 
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp.ascast, i64 0) 
  call spir_func void @Bar51(ptr addrspace(4) %Mask.ascast, ptr addrspace(4) align 4 %TmpMArray.ascast, ptr byval(%"range") %agg.tmp) 
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.end:                                          ; preds = %for.cond.cleanup
  ret void

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  %conv = sext i32 %1 to i64
  %call = call spir_func align 4 ptr addrspace(4) @Bar57(ptr addrspace(4) align 4 %TmpMArray.ascast, i64 %conv) 
  %2 = load i32, ptr addrspace(4) %call, align 4
  %3 = load i32, ptr %i, align 4
  %call1 = call spir_func align 4 ptr addrspace(4) @Bar20(ptr addrspace(4) align 16 %agg.result, i32 %3) 
  store i32 %2, ptr addrspace(4) %call1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
}


define internal spir_func align 4 ptr addrspace(4) @Bar20(ptr addrspace(4) align 16 %this, i32 %i) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %i.addr = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i32 %i, ptr %i.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %m_Data1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i32, ptr %i.addr, align 4
  %conv = sext i32 %0 to i64
  %call = call spir_func align 4 ptr addrspace(4) @_ZNSt5arrayIjLm4EEixEm(ptr addrspace(4) align 4 %m_Data1, i64 %conv) 
  ret ptr addrspace(4) %call
}


define internal spir_func align 4 ptr addrspace(4) @_ZNSt5arrayIjLm4EEixEm(ptr addrspace(4) align 4 %this, i64 %__n) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %__n.addr = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %__n, ptr %__n.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %_M_elems1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %__n.addr
  %call = call spir_func align 4 ptr addrspace(4) @_ZNSt14__array_traitsIjLm4EE6_S_refERA4_Kjm(ptr addrspace(4) align 4 %_M_elems1, i64 %0) 
  ret ptr addrspace(4) %call
}


define internal spir_func align 4 ptr addrspace(4) @_ZNSt14__array_traitsIjLm4EE6_S_refERA4_Kjm(ptr addrspace(4) align 4 %__t, i64 %__n) {
entry:
  %retval = alloca ptr addrspace(4)
  %__t.addr = alloca ptr addrspace(4)
  %__n.addr = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %__t, ptr %__t.addr
  store i64 %__n, ptr %__n.addr
  %0 = load ptr addrspace(4), ptr %__t.addr
  %1 = load i64, ptr %__n.addr
  %arrayidx = getelementptr inbounds nuw [4 x i32], ptr addrspace(4) %0, i64 0, i64 %1
  ret ptr addrspace(4) %arrayidx
}


define internal spir_func void @Bar50(ptr addrspace(4) align 4 %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = inttoptr i64 16 to ptr addrspace(4)
  br label %arrayinit.body

arrayinit.body:                                   ; preds = %arrayinit.body, %entry
  %lsr.iv = phi i64 [ %lsr.iv.next, %arrayinit.body ], [ 0, %entry ]
  %scevgep = getelementptr i8, ptr addrspace(4) %this1, i64 %lsr.iv
  store i32 0, ptr addrspace(4) %scevgep, align 4
  %lsr.iv.next = add nuw nsw i64 %lsr.iv, 4
  %lsr.iv.next1 = inttoptr i64 %lsr.iv.next to ptr addrspace(4)
  %arrayinit.done = icmp eq ptr addrspace(4) %lsr.iv.next1, %0
  br i1 %arrayinit.done, label %arrayinit.end2, label %arrayinit.body

arrayinit.end2:                                   ; preds = %arrayinit.body
  ret void
}


define internal spir_func void @Bar51(ptr addrspace(4) %this, ptr addrspace(4) align 4 %bits, ptr byval(%"range") %pos) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %bits.addr = alloca ptr addrspace(4)
  %cur_pos = alloca i64
  %__range4 = alloca ptr addrspace(4)
  %__begin0 = alloca ptr addrspace(4)
  %__end0 = alloca ptr addrspace(4)
  %cleanup.dest.slot = alloca i32, align 4
  %elem = alloca ptr addrspace(4)
  %agg.tmp = alloca %"range"
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %bits, ptr %bits.addr
  %pos.ascast = addrspacecast ptr %pos to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  %call = call spir_func i64 @Bar52(ptr addrspace(4) %pos.ascast, i32 0) 
  store i64 %call, ptr %cur_pos
  %0 = load ptr addrspace(4), ptr %bits.addr
  store ptr addrspace(4) %0, ptr %__range4
  %1 = load ptr addrspace(4), ptr %__range4
  %call2 = call spir_func ptr addrspace(4) @Bar53(ptr addrspace(4) align 4 %1) 
  store ptr addrspace(4) %call2, ptr %__begin0
  %2 = load ptr addrspace(4), ptr %__range4
  %call3 = call spir_func ptr addrspace(4) @Bar54(ptr addrspace(4) align 4 %2) 
  store ptr addrspace(4) %call3, ptr %__end0
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load ptr addrspace(4), ptr %__begin0
  %4 = load ptr addrspace(4), ptr %__end0
  %cmp = icmp ne ptr addrspace(4) %3, %4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.end:                                          ; preds = %for.cond.cleanup
  ret void

for.body:                                         ; preds = %for.cond
  %5 = load ptr addrspace(4), ptr %__begin0
  store ptr addrspace(4) %5, ptr %elem
  %6 = load i64, ptr %cur_pos
  %call4 = call spir_func i32 @Bar55(ptr addrspace(4) %this1) 
  %conv = zext i32 %call4 to i64
  %cmp5 = icmp ult i64 %6, %conv
  br i1 %cmp5, label %if.then, label %if.else

if.else:                                          ; preds = %for.body
  %7 = load ptr addrspace(4), ptr %elem
  store i32 0, ptr addrspace(4) %7, align 4
  br label %if.end

if.then:                                          ; preds = %for.body
  %8 = load ptr addrspace(4), ptr %elem
  %9 = load i64, ptr %cur_pos
  call spir_func void @Foo46(ptr addrspace(4) %agg.tmp.ascast, i64 %9) 
  call spir_func void @Bar56(ptr addrspace(4) %this1, ptr addrspace(4) align 4 %8, ptr byval(%"range") %agg.tmp) 
  %10 = load i64, ptr %cur_pos
  %add = add i64 %10, 32
  store i64 %add, ptr %cur_pos
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %11 = load ptr addrspace(4), ptr %__begin0
  %incdec.ptr = getelementptr inbounds nuw i32, ptr addrspace(4) %11, i32 1
  store ptr addrspace(4) %incdec.ptr, ptr %__begin0
  br label %for.cond
}


define internal spir_func align 4 ptr addrspace(4) @Bar57(ptr addrspace(4) align 4 %this, i64 %index) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %index.addr = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %index, ptr %index.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %MData1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %index.addr
  %arrayidx = getelementptr inbounds nuw [4 x i32], ptr addrspace(4) %MData1, i64 0, i64 %0
  ret ptr addrspace(4) %arrayidx
}


define internal spir_func i64 @Bar52(ptr addrspace(4) %this, i32 %dimension) {
entry:
  %this.addr.i = alloca ptr addrspace(4)
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %dimension.addr = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i32 %dimension, ptr %dimension.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i32, ptr %dimension.addr, align 4
  store ptr addrspace(4) %this1, ptr %this.addr.i
  store i32 %0, ptr %dimension.addr.i, align 4
  %this1.i = load ptr addrspace(4), ptr %this.addr.i
  %common_array1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load i32, ptr %dimension.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array1, i64 0, i64 %idxprom
  %2 = load i64, ptr addrspace(4) %arrayidx
  ret i64 %2
}


define internal spir_func ptr addrspace(4) @Bar53(ptr addrspace(4) align 4 %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr


  %this.addr1 = bitcast ptr %this.addr to ptr
  %this.addr2 = bitcast ptr %this.addr1 to ptr
  %this1 = load ptr addrspace(4), ptr %this.addr2



;  %this1 = load ptr addrspace(4), ptr %this.addr
  %MData1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %arraydecay2 = bitcast ptr addrspace(4) %MData1 to ptr addrspace(4)
  ret ptr addrspace(4) %arraydecay2
}


define internal spir_func ptr addrspace(4) @Bar54(ptr addrspace(4) align 4 %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr


  %this.addr1 = bitcast ptr %this.addr to ptr
  %this.addr2 = bitcast ptr %this.addr1 to ptr
  %this1 = load ptr addrspace(4), ptr %this.addr2

;  %this1 = load ptr addrspace(4), ptr %this.addr
  %MData1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %arraydecay2 = bitcast ptr addrspace(4) %MData1 to ptr addrspace(4)
  %add.ptr = getelementptr inbounds nuw i32, ptr addrspace(4) %arraydecay2, i64 4
  ret ptr addrspace(4) %add.ptr
}


define internal spir_func i32 @Bar55(ptr addrspace(4) %this) {
entry:
  %retval = alloca i32, align 4
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %bits_num = getelementptr inbounds nuw %"ss_sub_group_mask", ptr addrspace(4) %this1, i32 0, i32 1
  %0 = load i64, ptr addrspace(4) %bits_num
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}


define internal spir_func void @Bar56(ptr addrspace(4) %this, ptr addrspace(4) align 4 %bits, ptr byval(%"range") %pos) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %bits.addr = alloca ptr addrspace(4)
  %Res = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %bits, ptr %bits.addr
  %pos.ascast = addrspacecast ptr %pos to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  %Bits1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr addrspace(4) %Bits1
  store i64 %0, ptr %Res
  %bits_num = getelementptr inbounds nuw %"ss_sub_group_mask", ptr addrspace(4) %this1, i32 0, i32 1
  %1 = load i64, ptr addrspace(4) %bits_num
  %call = call spir_func i64 @Bar58(ptr addrspace(4) %this1, i64 %1) 
  %2 = load i64, ptr %Res
  %and = and i64 %2, %call
  store i64 %and, ptr %Res
  %call2 = call spir_func i64 @Bar52(ptr addrspace(4) %pos.ascast, i32 0) 
  %call3 = call spir_func i32 @Bar55(ptr addrspace(4) %this1) 
  %conv = zext i32 %call3 to i64
  %cmp = icmp ult i64 %call2, %conv
  br i1 %cmp, label %if.then, label %if.else

if.else:                                          ; preds = %entry
  %3 = load ptr addrspace(4), ptr %bits.addr
  store i32 0, ptr addrspace(4) %3, align 4
  br label %if.end11

if.then:                                          ; preds = %entry
  %call4 = call spir_func i64 @Bar52(ptr addrspace(4) %pos.ascast, i32 0) 
  %cmp5 = icmp ugt i64 %call4, 0
  br i1 %cmp5, label %if.then6, label %if.end

if.then6:                                         ; preds = %if.then
  %call7 = call spir_func i64 @Bar52(ptr addrspace(4) %pos.ascast, i32 0) 
  %4 = load i64, ptr %Res
  %shr = lshr i64 %4, %call7
  store i64 %shr, ptr %Res
  br label %if.end

if.end:                                           ; preds = %if.then6, %if.then
  %call8 = call spir_func i64 @Bar58(ptr addrspace(4) %this1, i64 32) 
  %5 = load i64, ptr %Res
  %and9 = and i64 %5, %call8
  store i64 %and9, ptr %Res
  %6 = load i64, ptr %Res
  %conv10 = trunc i64 %6 to i32
  %7 = load ptr addrspace(4), ptr %bits.addr
  store i32 %conv10, ptr addrspace(4) %7, align 4
  br label %if.end11

if.end11:                                         ; preds = %if.else, %if.end
  ret void
}


define internal spir_func i64 @Bar58(ptr addrspace(4) %this, i64 %bn) {
entry:
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %bn.addr = alloca i64
  %one = alloca i64
  %cleanup.dest.slot = alloca i32, align 4
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %bn, ptr %bn.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i64, ptr %bn.addr
  %cmp = icmp ule i64 %0, 64
  %1 = addrspacecast ptr addrspace(1) @.str.2 to ptr addrspace(4)
  %2 = addrspacecast ptr addrspace(1) @.str.1 to ptr addrspace(4)
  %3 = addrspacecast ptr addrspace(1) @__PRETTY_FUNCTION2 to ptr addrspace(4)
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  call spir_func void @__assert_fail(ptr addrspace(4) %1, ptr addrspace(4) %2, i32 327, ptr addrspace(4) %3) 
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.false
  store i64 1, ptr %one
  %4 = load i64, ptr %bn.addr
  %cmp2 = icmp eq i64 %4, 64
  br i1 %cmp2, label %if.then, label %if.end

if.end:                                           ; preds = %cond.end
  %5 = load i64, ptr %one
  %6 = load i64, ptr %bn.addr
  %shl = shl i64 %5, %6
  %7 = load i64, ptr %one
  %sub3 = sub i64 %shl, %7
  store i64 %sub3, ptr %retval
  store i32 1, ptr %cleanup.dest.slot, align 4
  br label %cleanup

if.then:                                          ; preds = %cond.end
  %8 = load i64, ptr %one
  %sub = sub i64 0, %8
  store i64 %sub, ptr %retval
  store i32 1, ptr %cleanup.dest.slot, align 4
  br label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %9 = load i64, ptr %retval
  ret i64 %9
}




define internal spir_func void @Foo11(ptr addrspace(4) %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @Foo60(ptr addrspace(4) %this1, i64 0) 
  ret void
}


define internal spir_func void @Foo60(ptr addrspace(4) %this, i64 %dim0) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %dim0.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %dim0, ptr %dim0.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %common_array1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %dim0.addr
  store i64 %0, ptr addrspace(4) %common_array1
  ret void
}


define internal spir_func i32 @Foo59(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"range") %local_id) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %LocalId = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %x.addr.ascast = addrspacecast ptr %x.addr to ptr addrspace(4)
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %call = call spir_func i32 @Foo48(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"range") %agg.tmp1) 
  store i32 %call, ptr %LocalId, align 4
  %0 = addrspacecast ptr addrspace(1) @_ZSt6ignore to ptr addrspace(4)
  %call2 = call spir_func align 1 ptr addrspace(4) @Bar15(ptr addrspace(4) align 1 %0, ptr addrspace(4) %g.ascast) 
  %call3 = call spir_func i32 @Foo49(ptr addrspace(4) align 4 %x.addr.ascast) 
  %1 = load i32, ptr %LocalId, align 4
  %call4 = call spir_func i32 @Foo50(i32 3, i32 %call3, i32 %1) 
  ret i32 %call4
}


define internal spir_func i32 @Foo47(ptr byval(%"ss_sub_group_mask") %Mask) {
entry:
  %retval = alloca i32, align 4
  %MemberMask = alloca %"vec.16", align 16
  %agg.tmp = alloca %"ss_sub_group_mask"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %MemberMask.ascast = addrspacecast ptr %MemberMask to ptr addrspace(4)
  call spir_func void @Bar19(ptr addrspace(4) dead_on_unwind writable sret(%"vec.16") align 16 %MemberMask.ascast, ptr byval(%"ss_sub_group_mask") %agg.tmp) 
  %call = call spir_func <4 x i32> @Bar59(ptr addrspace(4) align 16 %MemberMask.ascast) 
  %call1 = call spir_func i32 @_Z37__spirv_GroupNonUniformBallotBitCountN5__spv5Scope4FlagEiDv4_j(i32 3, i32 2, <4 x i32> %call) 
  ret i32 %call1
}


define internal spir_func <4 x i32> @Bar59(ptr addrspace(4) align 16 %x) {
entry:
  %retval = alloca <4 x i32>, align 16
  %x.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %x, ptr %x.addr
  %0 = load ptr addrspace(4), ptr %x.addr
  %call = call spir_func <4 x i32> @Bar60(ptr addrspace(4) align 16 %0) 
  ret <4 x i32> %call
}

declare dso_local spir_func i32 @_Z37__spirv_GroupNonUniformBallotBitCountN5__spv5Scope4FlagEiDv4_j(i32, i32, <4 x i32>) 


define internal spir_func <4 x i32> @Bar60(ptr addrspace(4) align 16 %from) {
entry:
  %retval = alloca <4 x i32>, align 16
  %from.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %from, ptr %from.addr
  %0 = load ptr addrspace(4), ptr %from.addr
  %1 = load <4 x i32>, ptr addrspace(4) %0, align 16
  ret <4 x i32> %1
}


define internal spir_func i32 @Foo52(ptr byval(%"tangle_group") %g, i32 %x, i32 %delta) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %delta.addr = alloca i32, align 4
  %TargetLocalId = alloca %"range"
  %TargetId = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp3 = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %x.addr.ascast = addrspacecast ptr %x.addr to ptr addrspace(4)
  %TargetLocalId.ascast = addrspacecast ptr %TargetLocalId to ptr addrspace(4)
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i32 %delta, ptr %delta.addr, align 4
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %TargetLocalId.ascast, ptr addrspace(4) %g.ascast) 
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %TargetLocalId.ascast, i32 0) 
  %0 = load i64, ptr addrspace(4) %call
  %1 = load i32, ptr %delta.addr, align 4
  %conv = zext i32 %1 to i64
  %cmp = icmp uge i64 %0, %conv
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %delta.addr, align 4
  %conv1 = zext i32 %2 to i64
  %call2 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %TargetLocalId.ascast, i32 0) 
  %3 = load i64, ptr addrspace(4) %call2
  %sub = sub i64 %3, %conv1
  store i64 %sub, ptr addrspace(4) %call2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %call4 = call spir_func i32 @Foo48(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"range") %agg.tmp3) 
  store i32 %call4, ptr %TargetId, align 4
  %call5 = call spir_func i32 @Foo49(ptr addrspace(4) align 4 %x.addr.ascast) 
  %4 = load i32, ptr %TargetId, align 4
  %call6 = call spir_func i32 @Foo50(i32 3, i32 %call5, i32 %4) 
  ret i32 %call6
}


define internal spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %this, i32 %dimension) {
entry:
  %this.addr.i = alloca ptr addrspace(4)
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %dimension.addr = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i32 %dimension, ptr %dimension.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i32, ptr %dimension.addr, align 4
  store ptr addrspace(4) %this1, ptr %this.addr.i
  store i32 %0, ptr %dimension.addr.i, align 4
  %this1.i = load ptr addrspace(4), ptr %this.addr.i
  %common_array1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load i32, ptr %dimension.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array1, i64 0, i64 %idxprom
  ret ptr addrspace(4) %arrayidx
}


define internal spir_func i32 @Foo72(ptr byval(%"tangle_group") %g, i32 %x, i32 %delta) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %delta.addr = alloca i32, align 4
  %TargetLocalId = alloca %"range"
  %TargetId = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp6 = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %x.addr.ascast = addrspacecast ptr %x.addr to ptr addrspace(4)
  %TargetLocalId.ascast = addrspacecast ptr %TargetLocalId to ptr addrspace(4)
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i32 %delta, ptr %delta.addr, align 4
  call spir_func void @Foo51(ptr addrspace(4) dead_on_unwind writable sret(%"range") %TargetLocalId.ascast, ptr addrspace(4) %g.ascast) 
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %TargetLocalId.ascast, i32 0) 
  %0 = load i64, ptr addrspace(4) %call
  %1 = load i32, ptr %delta.addr, align 4
  %conv = zext i32 %1 to i64
  %add = add i64 %0, %conv
  %call1 = call spir_func i32 @Bar61(ptr addrspace(4) %g.ascast) 
  %conv2 = zext i32 %call1 to i64
  %cmp = icmp ult i64 %add, %conv2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %delta.addr, align 4
  %conv3 = zext i32 %2 to i64
  %call4 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %TargetLocalId.ascast, i32 0) 
  %3 = load i64, ptr addrspace(4) %call4
  %add5 = add i64 %3, %conv3
  store i64 %add5, ptr addrspace(4) %call4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %call7 = call spir_func i32 @Foo48(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"range") %agg.tmp6) 
  store i32 %call7, ptr %TargetId, align 4
  %call8 = call spir_func i32 @Foo49(ptr addrspace(4) align 4 %x.addr.ascast) 
  %4 = load i32, ptr %TargetId, align 4
  %call9 = call spir_func i32 @Foo50(i32 3, i32 %call8, i32 %4) 
  ret i32 %call9
}


define internal spir_func i32 @Bar61(ptr addrspace(4) %this) {
entry:
  %retval = alloca i32, align 4
  %this.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @Foo97(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) %this1) 
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %ref.tmp.ascast, i32 0) 
  %0 = load i64, ptr addrspace(4) %call
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}


define internal spir_func void @Foo97(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %Mask1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %call = call spir_func i32 @Bar62(ptr addrspace(4) %Mask1) 
  %conv = zext i32 %call to i64
  call spir_func void @Foo9(ptr addrspace(4) %agg.result, i64 %conv) 
  ret void
}


define internal spir_func i32 @Bar62(ptr addrspace(4) %this) {
entry:
  %retval = alloca i32, align 4
  %this.addr = alloca ptr addrspace(4)
  %count = alloca i32, align 4
  %word = alloca i64
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  store i32 0, ptr %count, align 4
  %Bits1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr addrspace(4) %Bits1
  %bits_num = getelementptr inbounds nuw %"ss_sub_group_mask", ptr addrspace(4) %this1, i32 0, i32 1
  %1 = load i64, ptr addrspace(4) %bits_num
  %call = call spir_func i64 @Bar58(ptr addrspace(4) %this1, i64 %1) 
  %and = and i64 %0, %call
  store i64 %and, ptr %word
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %2 = load i64, ptr %word
  %tobool = icmp ne i64 %2, 0
  br i1 %tobool, label %while.body, label %while.end

while.end:                                        ; preds = %while.cond
  %3 = load i32, ptr %count, align 4
  ret i32 %3

while.body:                                       ; preds = %while.cond
  %4 = load i64, ptr %word
  %sub = sub i64 %4, 1
  %5 = load i64, ptr %word
  %and2 = and i64 %5, %sub
  store i64 %and2, ptr %word
  %6 = load i32, ptr %count, align 4
  %inc = add i32 %6, 1
  store i32 %inc, ptr %count, align 4
  br label %while.cond
}


define internal spir_func void @Foo9(ptr addrspace(4) %this, i64 %dim0) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  %dim0.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %dim0, ptr %dim0.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i64, ptr %dim0.addr
  call spir_func void @Foo60(ptr addrspace(4) %this1, i64 %0) 
  ret void
}


define internal spir_func i32 @Foo61(ptr byval(%"tangle_group") %g, ptr byval(%"nd_item") align 1 %0, i32 %x, ptr byval(%"nd_item") align 1 %1){
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"nd_item", align 1
  %agg.tmp2 = alloca %"nd_item", align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar63(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"nd_item") align 1 %agg.tmp1, i32 %4, ptr byval(%"nd_item") align 1 %agg.tmp2) 
  ret i32 %call
}


define internal spir_func i32 @Bar63(ptr byval(%"tangle_group") %g, ptr byval(%"nd_item") align 1 %0, i32 %x, ptr byval(%"nd_item") align 1 %1){
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar64(ptr byval(%"tangle_group") %agg.tmp, i32 %4) 
  ret i32 %call
}


define internal spir_func i32 @Bar64(ptr byval(%"tangle_group") %0, i32 %x)   {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %Arg = alloca i32, align 4
  %Ret = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %2 = load i32, ptr %x.addr, align 4
  store i32 %2, ptr %Arg, align 4
  %3 = load i32, ptr %Arg, align 4
  %call = call spir_func i32 @_Z27__spirv_GroupNonUniformIAddIiET_N5__spv5Scope4FlagEjS0_(i32 3, i32 1, i32 %3) 
  store i32 %call, ptr %Ret, align 4
  %4 = load i32, ptr %Ret, align 4
  ret i32 %4
}

declare dso_local spir_func i32 @_Z27__spirv_GroupNonUniformIAddIiET_N5__spv5Scope4FlagEjS0_(i32, i32, i32) 


define internal spir_func i32 @Bar12(ptr byval(%"tangle_group") %g, ptr byval(%"nd_item") align 1 %0, i32 %x, ptr byval(%"nd_item") align 1 %1){
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"nd_item", align 1
  %agg.tmp2 = alloca %"nd_item", align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar65(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"nd_item") align 1 %agg.tmp1, i32 %4, ptr byval(%"nd_item") align 1 %agg.tmp2) 
  ret i32 %call
}


define internal spir_func i32 @Bar65(ptr byval(%"tangle_group") %g, ptr byval(%"nd_item") align 1 %0, i32 %x, ptr byval(%"nd_item") align 1 %1){
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar66(ptr byval(%"tangle_group") %agg.tmp, i32 %4) 
  ret i32 %call
}


define internal spir_func i32 @Bar66(ptr byval(%"tangle_group") %0, i32 %x)   {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %Arg = alloca i32, align 4
  %Ret = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %2 = load i32, ptr %x.addr, align 4
  store i32 %2, ptr %Arg, align 4
  %3 = load i32, ptr %Arg, align 4
  %call = call spir_func i32 @_Z27__spirv_GroupNonUniformIAddIiET_N5__spv5Scope4FlagEjS0_(i32 3, i32 2, i32 %3) 
  store i32 %call, ptr %Ret, align 4
  %4 = load i32, ptr %Ret, align 4
  ret i32 %4
}


define internal spir_func i32 @Bar11(ptr byval(%"tangle_group") %g, ptr byval(%"nd_item") align 1 %0, i32 %x, ptr byval(%"nd_item") align 1 %1){
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"nd_item", align 1
  %agg.tmp2 = alloca %"nd_item", align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar67(ptr byval(%"tangle_group") %agg.tmp, ptr byval(%"nd_item") align 1 %agg.tmp1, i32 %4, ptr byval(%"nd_item") align 1 %agg.tmp2) 
  ret i32 %call
}


define internal spir_func i32 @Bar67(ptr byval(%"tangle_group") %g, ptr byval(%"nd_item") align 1 %0, i32 %x, ptr byval(%"nd_item") align 1 %1){
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar68(ptr byval(%"tangle_group") %agg.tmp, i32 %4) 
  ret i32 %call
}


define internal spir_func i32 @Bar68(ptr byval(%"tangle_group") %0, i32 %x)   {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %Arg = alloca i32, align 4
  %Ret = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %2 = load i32, ptr %x.addr, align 4
  store i32 %2, ptr %Arg, align 4
  %3 = load i32, ptr %Arg, align 4
  %call = call spir_func i32 @_Z27__spirv_GroupNonUniformIAddIiET_N5__spv5Scope4FlagEjS0_(i32 3, i32 0, i32 %3) 
  store i32 %call, ptr %Ret, align 4
  %4 = load i32, ptr %Ret, align 4
  ret i32 %4
}


define internal spir_func zeroext i1 @Foo66(ptr byval(%"tangle_group") %0, i1 zeroext %pred) {
entry:
  %retval = alloca i1, align 1
  %pred.addr = alloca i8, align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %storedv = zext i1 %pred to i8
  store i8 %storedv, ptr %pred.addr, align 1
  %2 = load i8, ptr %pred.addr, align 1  
  %loadedv = trunc i8 %2 to i1
  %call = call spir_func zeroext i1 @Foo99(i32 3, i1 zeroext %loadedv) 
  ret i1 %call
}

declare dso_local spir_func zeroext i1 @Foo99(i32, i1 zeroext) 


define internal spir_func zeroext i1 @Bar10(ptr byval(%"tangle_group") %0, i1 zeroext %pred) {
entry:
  %retval = alloca i1, align 1
  %pred.addr = alloca i8, align 1
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %storedv = zext i1 %pred to i8
  store i8 %storedv, ptr %pred.addr, align 1
  %2 = load i8, ptr %pred.addr, align 1  
  %loadedv = trunc i8 %2 to i1
  %call = call spir_func zeroext i1 @_Z26__spirv_GroupNonUniformAnyN5__spv5Scope4FlagEb(i32 3, i1 zeroext %loadedv) 
  ret i1 %call
}

declare dso_local spir_func zeroext i1 @_Z26__spirv_GroupNonUniformAnyN5__spv5Scope4FlagEb(i32, i1 zeroext) 


define internal spir_func void @Foo98(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr byval(%"range") %0, i64 %linear_id)   {
entry:
  %linear_id.addr = alloca i64
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store i64 %linear_id, ptr %linear_id.addr
  %2 = load i64, ptr %linear_id.addr
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %2) 
  ret void
}


define internal spir_func i32 @Bar69(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"range") %local_id) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %agg.tmp1 = alloca %"range"
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @Bar70(ptr byval(%"tangle_group") %agg.tmp, i32 %0, ptr byval(%"range") %agg.tmp1) 
  ret i32 %call
}


define internal spir_func i32 @Bar70(ptr byval(%"tangle_group") %g, i32 %x, ptr byval(%"range") %local_id) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %VecId = alloca %"range"
  %OCLX = alloca i32, align 4
  %WideOCLX = alloca i32, align 4
  %OCLId = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %VecId.ascast = addrspacecast ptr %VecId to ptr addrspace(4)
  %OCLX.ascast = addrspacecast ptr %OCLX to ptr addrspace(4)
  %WideOCLX.ascast = addrspacecast ptr %WideOCLX to ptr addrspace(4)
  %OCLId.ascast = addrspacecast ptr %OCLId to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  %local_id.ascast = addrspacecast ptr %local_id to ptr addrspace(4)
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %local_id.ascast, i32 0) 
  %1 = load i64, ptr addrspace(4) %call
  %call1 = call spir_func i32 @Bar71(ptr byval(%"tangle_group") %agg.tmp, i32 %0, i64 %1) 
  ret i32 %call1
}


define internal spir_func i32 @Bar71(ptr byval(%"tangle_group") %g, i32 %x, i64 %local_id) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %local_id.addr = alloca i64
  %LocalId = alloca i32, align 4
  %agg.tmp = alloca %"tangle_group"
  %GroupLocalId = alloca i32, align 4
  %OCLX = alloca i32, align 4
  %WideOCLX = alloca i32, align 4
  %OCLId = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %x.addr.ascast = addrspacecast ptr %x.addr to ptr addrspace(4)
  %GroupLocalId.ascast = addrspacecast ptr %GroupLocalId to ptr addrspace(4)
  store i32 %x, ptr %x.addr, align 4
  store i64 %local_id, ptr %local_id.addr
  %0 = load i64, ptr %local_id.addr
  %conv = trunc i64 %0 to i32
  %call = call spir_func i32 @Bar17(ptr byval(%"tangle_group") %agg.tmp, i32 %conv) 
  store i32 %call, ptr %LocalId, align 4
  %1 = load i32, ptr %LocalId, align 4
  store i32 %1, ptr %GroupLocalId, align 4
  %call1 = call spir_func i32 @Foo49(ptr addrspace(4) align 4 %x.addr.ascast) 
  store i32 %call1, ptr %OCLX, align 4
  %2 = load i32, ptr %OCLX, align 4
  store i32 %2, ptr %WideOCLX, align 4
  %call2 = call spir_func i32 @Foo49(ptr addrspace(4) align 4 %GroupLocalId.ascast) 
  store i32 %call2, ptr %OCLId, align 4
  %3 = load i32, ptr %WideOCLX, align 4
  %4 = load i32, ptr %OCLId, align 4
  %call3 = call spir_func i32 @_Z32__spirv_GroupNonUniformBroadcastIjjET_N5__spv5Scope4FlagES0_T0_(i32 3, i32 %3, i32 %4) 
  ret i32 %call3
}

declare dso_local spir_func i32 @_Z32__spirv_GroupNonUniformBroadcastIjjET_N5__spv5Scope4FlagES0_T0_(i32, i32, i32) 


define internal spir_func void @Foo96(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) align 1 %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %call = call spir_func i32 @_Z33__spirv_SubgroupLocalInvocationIdv() 
  %conv = zext i32 %call to i64
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %conv) 
  ret void
}


define internal spir_func i32 @_Z33__spirv_SubgroupLocalInvocationIdv()   {
entry:
  %retval = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load i32, ptr addrspace(1) @__spirv_BuiltInSubgroupLocalInvocationId, align 4
  ret i32 %0
}


define internal spir_func i64 @Foo77(ptr addrspace(4) %this, ptr byval(%"range") %Id) {
entry:
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %Result = alloca i64
  %ref.tmp = alloca %class.anon.15
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %Result.ascast = addrspacecast ptr %Result to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %Id.ascast = addrspacecast ptr %Id to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  store i64 0, ptr %Result
  %0 = bitcast ptr %ref.tmp to ptr
  store ptr addrspace(4) %this1, ptr %0
  %Result2 = getelementptr inbounds %class.anon.15, ptr %ref.tmp, i32 0, i32 1
  store ptr addrspace(4) %Result.ascast, ptr %Result2
  %Id3 = getelementptr inbounds %class.anon.15, ptr %ref.tmp, i32 0, i32 2
  store ptr addrspace(4) %Id.ascast, ptr %Id3
  call spir_func void @Foo79(ptr addrspace(4) %ref.tmp.ascast) 
  %1 = load i64, ptr %Result
  ret i64 %1
}


define internal spir_func ptr addrspace(1) @Foo78(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(1)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = getelementptr inbounds nuw %"accessor", ptr addrspace(4) %this1, i32 0, i32 1
  %1 = load ptr addrspace(1), ptr addrspace(4) %0
  ret ptr addrspace(1) %1
}


define internal spir_func void @Foo79(ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"nd_item", align 1
  store ptr addrspace(4) %f, ptr %f.addr
  %0 = load ptr addrspace(4), ptr %f.addr
  call spir_func void @Foo80(ptr byval(%"nd_item") align 1 %agg.tmp, ptr addrspace(4) %0) 
  ret void
}


define internal spir_func void @Foo80(ptr byval(%"nd_item") align 1 %0, ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"nd_item", align 1
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store ptr addrspace(4) %f, ptr %f.addr
  %2 = load ptr addrspace(4), ptr %f.addr
  %call = call spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %ref.tmp.ascast) 
  call spir_func void @Foo81(ptr addrspace(4) %2, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %this) {
entry:
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  ret i64 0
}


define internal spir_func void @Foo81(ptr addrspace(4) %this, i64 %I)  align 2  {
entry:
  %this.addr = alloca ptr addrspace(4)
  %I.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load ptr addrspace(4), ptr addrspace(4) %0
  %Result = getelementptr inbounds nuw %class.anon.15, ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load ptr addrspace(4), ptr addrspace(4) %Result
  %3 = load i64, ptr addrspace(4) %2
  %call = call spir_func ptr addrspace(4) @Bar72(ptr addrspace(4) %1) 
  %4 = load i64, ptr %I.addr
  %conv = trunc i64 %4 to i32
  %call2 = call spir_func i64 @Foo37(ptr addrspace(4) %call, i32 %conv) 
  %mul = mul i64 %3, %call2
  %Id = getelementptr inbounds nuw %class.anon.15, ptr addrspace(4) %this1, i32 0, i32 2
  %5 = load ptr addrspace(4), ptr addrspace(4) %Id
  %6 = load i64, ptr %I.addr
  %conv3 = trunc i64 %6 to i32
  %call4 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %5, i32 %conv3) 
  %7 = load i64, ptr addrspace(4) %call4
  %add = add i64 %mul, %7
  %Result5 = getelementptr inbounds nuw %class.anon.15, ptr addrspace(4) %this1, i32 0, i32 1
  %8 = load ptr addrspace(4), ptr addrspace(4) %Result5
  store i64 %add, ptr addrspace(4) %8
  ret void
}


define internal spir_func ptr addrspace(4) @Bar72(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 2
  ret ptr addrspace(4) %MemRange
}


define internal spir_func i64 @Foo37(ptr addrspace(4) %this, i32 %dimension) {
entry:
  %this.addr.i = alloca ptr addrspace(4)
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %dimension.addr = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store i32 %dimension, ptr %dimension.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i32, ptr %dimension.addr, align 4
  store ptr addrspace(4) %this1, ptr %this.addr.i
  store i32 %0, ptr %dimension.addr.i, align 4
  %this1.i = load ptr addrspace(4), ptr %this.addr.i
  %common_array1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load i32, ptr %dimension.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array1, i64 0, i64 %idxprom
  %2 = load i64, ptr addrspace(4) %arrayidx
  ret i64 %2
}


define internal spir_func void @Foo95(ptr byval(%"tangle_group") %g, i32 %FenceScope, i32 %Order) {
entry:
  %FenceScope.addr = alloca i32, align 4
  %Order.addr = alloca i32, align 4
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  store i32 %FenceScope, ptr %FenceScope.addr, align 4
  store i32 %Order, ptr %Order.addr, align 4
  %0 = load i32, ptr %FenceScope.addr, align 4
  %call = call spir_func i32 @Bar73(i32 %0) 
  %1 = load i32, ptr %Order.addr, align 4
  %call1 = call spir_func i32 @Bar74(i32 %1) 
  %or = or i32 %call1, 128
  %or2 = or i32 %or, 256
  %or3 = or i32 %or2, 512
  call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 %call, i32 %or3) 
  ret void
}


define internal spir_func i32 @Bar73(i32 %Scope){
entry:
  %retval = alloca i32, align 4
  %Scope.addr = alloca i32, align 4
  store i32 %Scope, ptr %Scope.addr, align 4
  %0 = load i32, ptr %Scope.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
  ]

sw.bb4:                                           ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

sw.bb3:                                           ; preds = %entry
  store i32 1, ptr %retval, align 4
  br label %return

sw.bb2:                                           ; preds = %entry
  store i32 2, ptr %retval, align 4
  br label %return

sw.bb1:                                           ; preds = %entry
  store i32 3, ptr %retval, align 4
  br label %return

sw.bb:                                            ; preds = %entry
  store i32 4, ptr %retval, align 4
  br label %return

return:                                           ; preds = %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %1 = load i32, ptr %retval, align 4
  ret i32 %1

sw.epilog:                                        ; preds = %entry
  unreachable
}


define internal spir_func i32 @Bar74(i32 %Order){
entry:
  %retval = alloca i32, align 4
  %Order.addr = alloca i32, align 4
  %SpvOrder = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 %Order, ptr %Order.addr, align 4
  store i32 0, ptr %SpvOrder, align 4
  %0 = load i32, ptr %Order.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 2, label %sw.bb1
    i32 1, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb3
    i32 5, label %sw.bb4
  ]

sw.bb4:                                           ; preds = %entry
  store i32 16, ptr %SpvOrder, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  store i32 8, ptr %SpvOrder, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  store i32 4, ptr %SpvOrder, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry, %entry
  store i32 2, ptr %SpvOrder, align 4
  br label %sw.epilog

sw.bb:                                            ; preds = %entry
  store i32 0, ptr %SpvOrder, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb, %entry
  %1 = load i32, ptr %SpvOrder, align 4
  %or = or i32 %1, 128
  %or5 = or i32 %or, 256
  %or6 = or i32 %or5, 512
  ret i32 %or6
}

declare dso_local spir_func void @_Z21__spirv_MemoryBarrierjj(i32, i32) 


define internal spir_func i64 @Foo93(ptr addrspace(4) %this, ptr byval(%"range") %Id) {
entry:
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %Result = alloca i64
  %ref.tmp = alloca %class.anon.15
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %Result.ascast = addrspacecast ptr %Result to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %Id.ascast = addrspacecast ptr %Id to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr
  store i64 0, ptr %Result
  %0 = bitcast ptr %ref.tmp to ptr
  store ptr addrspace(4) %this1, ptr %0
  %Result2 = getelementptr inbounds %class.anon.15, ptr %ref.tmp, i32 0, i32 1
  store ptr addrspace(4) %Result.ascast, ptr %Result2
  %Id3 = getelementptr inbounds %class.anon.15, ptr %ref.tmp, i32 0, i32 2
  store ptr addrspace(4) %Id.ascast, ptr %Id3
  call spir_func void @Bar75(ptr addrspace(4) %ref.tmp.ascast) 
  %1 = load i64, ptr %Result
  ret i64 %1
}


define internal spir_func ptr addrspace(1) @Foo94(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(1)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = getelementptr inbounds nuw %"accessor", ptr addrspace(4) %this1, i32 0, i32 1
  %1 = load ptr addrspace(1), ptr addrspace(4) %0
  ret ptr addrspace(1) %1
}


define internal spir_func void @Bar75(ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"nd_item", align 1
  store ptr addrspace(4) %f, ptr %f.addr
  %0 = load ptr addrspace(4), ptr %f.addr
  call spir_func void @Bar76(ptr byval(%"nd_item") align 1 %agg.tmp, ptr addrspace(4) %0) 
  ret void
}


define internal spir_func void @Bar76(ptr byval(%"nd_item") align 1 %0, ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"nd_item", align 1
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store ptr addrspace(4) %f, ptr %f.addr
  %2 = load ptr addrspace(4), ptr %f.addr
  %call = call spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %ref.tmp.ascast) 
  call spir_func void @Bar767(ptr addrspace(4) %2, i64 %call) 
  ret void
}


define internal spir_func void @Bar767(ptr addrspace(4) %this, i64 %I)  align 2  {
entry:
  %this.addr = alloca ptr addrspace(4)
  %I.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load ptr addrspace(4), ptr addrspace(4) %0
  %Result = getelementptr inbounds nuw %class.anon.15, ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load ptr addrspace(4), ptr addrspace(4) %Result
  %3 = load i64, ptr addrspace(4) %2
  %call = call spir_func ptr addrspace(4) @Bar78(ptr addrspace(4) %1) 
  %4 = load i64, ptr %I.addr
  %conv = trunc i64 %4 to i32
  %call2 = call spir_func i64 @Foo37(ptr addrspace(4) %call, i32 %conv) 
  %mul = mul i64 %3, %call2
  %Id = getelementptr inbounds nuw %class.anon.15, ptr addrspace(4) %this1, i32 0, i32 2
  %5 = load ptr addrspace(4), ptr addrspace(4) %Id
  %6 = load i64, ptr %I.addr
  %conv3 = trunc i64 %6 to i32
  %call4 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %5, i32 %conv3) 
  %7 = load i64, ptr addrspace(4) %call4
  %add = add i64 %mul, %7
  %Result5 = getelementptr inbounds nuw %class.anon.15, ptr addrspace(4) %this1, i32 0, i32 1
  %8 = load ptr addrspace(4), ptr addrspace(4) %Result5
  store i64 %add, ptr addrspace(4) %8
  ret void
}


define internal spir_func ptr addrspace(4) @Bar78(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 2
  ret ptr addrspace(4) %MemRange
}


define internal spir_func void @Foo44(ptr addrspace(4) dead_on_unwind noalias writable sret(%"ss_sub_group_mask") %agg.result, ptr byval(%"nd_item") align 1 %g, i1 zeroext %predicate) {
entry:
  %predicate.addr = alloca i8, align 1
  %res = alloca <4 x i32>, align 16
  %val = alloca i64
  %ref.tmp = alloca %"range"
  %cleanup.dest.slot = alloca i32, align 4
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %g.ascast = addrspacecast ptr %g to ptr addrspace(4)
  %storedv = zext i1 %predicate to i8
  store i8 %storedv, ptr %predicate.addr, align 1
  %0 = load i8, ptr %predicate.addr, align 1  
  %loadedv = trunc i8 %0 to i1
  %call = call spir_func <4 x i32> @_Z29__spirv_GroupNonUniformBallotjb(i32 3, i1 zeroext %loadedv) 
  store <4 x i32> %call, ptr %res, align 16
  %1 = load <4 x i32>, ptr %res, align 16
  %vecext = extractelement <4 x i32> %1, i32 0
  %conv = zext i32 %vecext to i64
  store i64 %conv, ptr %val
  %2 = load <4 x i32>, ptr %res, align 16
  %vecext1 = extractelement <4 x i32> %2, i32 1
  %conv2 = zext i32 %vecext1 to i64
  %shl = shl i64 %conv2, 32
  %3 = load i64, ptr %val
  %or = or i64 %3, %shl
  store i64 %or, ptr %val
  %4 = load i64, ptr %val
  call spir_func void @Bar79(ptr addrspace(4) dead_on_unwind writable sret(%"range") %ref.tmp.ascast, ptr addrspace(4) align 1 %g.ascast) 
  %call3 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %ref.tmp.ascast, i32 0) 
  %5 = load i64, ptr addrspace(4) %call3
  call spir_func void @Bar80(ptr addrspace(4) dead_on_unwind writable sret(%"ss_sub_group_mask") %agg.result, i64 %4, i64 %5) 
  ret void
}


define internal spir_func void @Foo45(ptr addrspace(4) %this, ptr byval(%"ss_sub_group_mask") %m) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %Mask1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  ret void
}

declare dso_local spir_func <4 x i32> @_Z29__spirv_GroupNonUniformBallotjb(i32, i1 zeroext) 


define internal spir_func void @Bar79(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result, ptr addrspace(4) align 1 %this) {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %call = call spir_func i32 @_Z23__spirv_SubgroupMaxSizev() 
  %conv = zext i32 %call to i64
  call spir_func void @Foo9(ptr addrspace(4) %agg.result, i64 %conv) 
  ret void
}


define internal spir_func void @Bar80(ptr addrspace(4) dead_on_unwind noalias writable sret(%"ss_sub_group_mask") %agg.result, i64 %Bits, i64 %BitsNum) {
entry:
  %Bits.addr = alloca i64
  %BitsNum.addr = alloca i64
  store i64 %Bits, ptr %Bits.addr
  store i64 %BitsNum, ptr %BitsNum.addr
  %0 = load i64, ptr %Bits.addr
  %1 = load i64, ptr %BitsNum.addr
  call spir_func void @Bar81(ptr addrspace(4) %agg.result, i64 %0, i64 %1) 
  ret void
}


define internal spir_func void @Bar81(ptr addrspace(4) %this, i64 %rhs, i64 %bn) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  %rhs.addr = alloca i64
  %bn.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %rhs, ptr %rhs.addr
  store i64 %bn, ptr %bn.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %Bits1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %rhs.addr
  %1 = load i64, ptr %bn.addr
  %call = call spir_func i64 @Bar58(ptr addrspace(4) %this1, i64 %1) 
  %and = and i64 %0, %call
  store i64 %and, ptr addrspace(4) %Bits1
  %bits_num = getelementptr inbounds nuw %"ss_sub_group_mask", ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load i64, ptr %bn.addr
  store i64 %2, ptr addrspace(4) %bits_num
  %bits_num2 = getelementptr inbounds nuw %"ss_sub_group_mask", ptr addrspace(4) %this1, i32 0, i32 1
  %3 = load i64, ptr addrspace(4) %bits_num2
  %cmp = icmp ule i64 %3, 64
  %4 = addrspacecast ptr addrspace(1) @.str to ptr addrspace(4)
  %5 = addrspacecast ptr addrspace(1) @.str.1 to ptr addrspace(4)
  %6 = addrspacecast ptr addrspace(1) @__PRETTY_FUNCTION1 to ptr addrspace(4)
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  call spir_func void @__assert_fail(ptr addrspace(4) %4, ptr addrspace(4) %5, i32 324, ptr addrspace(4) %6) 
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.false
  ret void
}


define internal spir_func i32 @_Z23__spirv_SubgroupMaxSizev()   {
entry:
  %retval = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load i32, ptr addrspace(1) @__spirv_BuiltInSubgroupMaxSize, align 4
  ret i32 %0
}


define internal spir_func void @Init6(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @Inv1(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Inv1(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @Inv2() 
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @Inv2() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @Foo7(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @Foo8(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Init1(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @Inv3(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Init2(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @InitSize1(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Init3(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @InitSize2(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Init4(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @InitSize3(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Init5(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @InitSize4(ptr addrspace(4) dead_on_unwind writable sret(%"range") %agg.result) 
  ret void
}


define internal spir_func void @Foo23(ptr addrspace(4) dead_on_unwind noalias writable sret(%"group") %agg.result, ptr addrspace(4) %Global, ptr addrspace(4) %Local, ptr addrspace(4) %Group, ptr addrspace(4) %Index) {
entry:
  %Global.addr = alloca ptr addrspace(4)
  %Local.addr = alloca ptr addrspace(4)
  %Group.addr = alloca ptr addrspace(4)
  %Index.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"range"
  store ptr addrspace(4) %Global, ptr %Global.addr
  store ptr addrspace(4) %Local, ptr %Local.addr
  store ptr addrspace(4) %Group, ptr %Group.addr
  store ptr addrspace(4) %Index, ptr %Index.addr
  %0 = load ptr addrspace(4), ptr %Global.addr
  %1 = load ptr addrspace(4), ptr %Local.addr
  %2 = load ptr addrspace(4), ptr %Group.addr
  %3 = load ptr addrspace(4), ptr %Index.addr
  call spir_func void @Bar82(ptr addrspace(4) %agg.result, ptr addrspace(4) %0, ptr addrspace(4) %1, ptr byval(%"range") %agg.tmp, ptr addrspace(4) %3) 
  ret void
}


define internal spir_func void @Foo24(ptr addrspace(4) dead_on_unwind noalias writable sret(%"item") %agg.result, ptr addrspace(4) %Extent, ptr addrspace(4) %Index, ptr addrspace(4) %Offset) {
entry:
  %Extent.addr = alloca ptr addrspace(4)
  %Index.addr = alloca ptr addrspace(4)
  %Offset.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %Extent, ptr %Extent.addr
  store ptr addrspace(4) %Index, ptr %Index.addr
  store ptr addrspace(4) %Offset, ptr %Offset.addr
  %0 = load ptr addrspace(4), ptr %Extent.addr
  %1 = load ptr addrspace(4), ptr %Index.addr
  %2 = load ptr addrspace(4), ptr %Offset.addr
  call spir_func void @Foo29(ptr addrspace(4) %agg.result, ptr addrspace(4) %0, ptr addrspace(4) %1, ptr addrspace(4) %2) 
  ret void
}


define internal spir_func void @Foo25(ptr addrspace(4) dead_on_unwind noalias writable sret(%"item.22") %agg.result, ptr addrspace(4) %Extent, ptr addrspace(4) %Index) {
entry:
  %Extent.addr = alloca ptr addrspace(4)
  %Index.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %Extent, ptr %Extent.addr
  store ptr addrspace(4) %Index, ptr %Index.addr
  %0 = load ptr addrspace(4), ptr %Extent.addr
  %1 = load ptr addrspace(4), ptr %Index.addr
  call spir_func void @Foo27(ptr addrspace(4) %agg.result, ptr addrspace(4) %0, ptr addrspace(4) %1) 
  ret void
}


define internal spir_func void @Foo26(ptr addrspace(4) dead_on_unwind noalias writable sret(%"nd_item") align 1 %agg.result, ptr addrspace(4) %Global, ptr addrspace(4) %Local, ptr addrspace(4) %Group) {
entry:
  %Global.addr = alloca ptr addrspace(4)
  %Local.addr = alloca ptr addrspace(4)
  %Group.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %Global, ptr %Global.addr
  store ptr addrspace(4) %Local, ptr %Local.addr
  store ptr addrspace(4) %Group, ptr %Group.addr
  %0 = load ptr addrspace(4), ptr %Global.addr
  %1 = load ptr addrspace(4), ptr %Local.addr
  %2 = load ptr addrspace(4), ptr %Group.addr
  call spir_func void @Foo28(ptr addrspace(4) align 1 %agg.result, ptr addrspace(4) %0, ptr addrspace(4) %1, ptr addrspace(4) %2) 
  ret void
}


define internal spir_func void @Foo28(ptr addrspace(4) align 1 %this, ptr addrspace(4) %0, ptr addrspace(4) %1, ptr addrspace(4) %2) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  %.addr = alloca ptr addrspace(4)
  %.addr1 = alloca ptr addrspace(4)
  %.addr2 = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %0, ptr %.addr
  store ptr addrspace(4) %1, ptr %.addr1
  store ptr addrspace(4) %2, ptr %.addr2
  %this3 = load ptr addrspace(4), ptr %this.addr
  ret void
}


define internal spir_func void @Foo27(ptr addrspace(4) %this, ptr addrspace(4) %extent, ptr addrspace(4) %index) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  %extent.addr = alloca ptr addrspace(4)
  %index.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %extent, ptr %extent.addr
  store ptr addrspace(4) %index, ptr %index.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %MImpl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %MExtent2 = bitcast ptr addrspace(4) %MImpl1 to ptr addrspace(4)
  %0 = load ptr addrspace(4), ptr %extent.addr
  %MIndex = getelementptr inbounds nuw %"sd_ItemBase.23", ptr addrspace(4) %MImpl1, i32 0, i32 1
  %1 = load ptr addrspace(4), ptr %index.addr
  ret void
}




define internal spir_func void @Foo29(ptr addrspace(4) %this, ptr addrspace(4) %extent, ptr addrspace(4) %index, ptr addrspace(4) %offset) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  %extent.addr = alloca ptr addrspace(4)
  %index.addr = alloca ptr addrspace(4)
  %offset.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %extent, ptr %extent.addr
  store ptr addrspace(4) %index, ptr %index.addr
  store ptr addrspace(4) %offset, ptr %offset.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %MImpl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %MExtent2 = bitcast ptr addrspace(4) %MImpl1 to ptr addrspace(4)
  %0 = load ptr addrspace(4), ptr %extent.addr
  %MIndex = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %MImpl1, i32 0, i32 1
  %1 = load ptr addrspace(4), ptr %index.addr
  %MOffset = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %MImpl1, i32 0, i32 2
  %2 = load ptr addrspace(4), ptr %offset.addr
  ret void
}


define internal spir_func void @Bar82(ptr addrspace(4) %this, ptr addrspace(4) %G, ptr addrspace(4) %L, ptr byval(%"range") %GroupRange, ptr addrspace(4) %I) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  %G.addr = alloca ptr addrspace(4)
  %L.addr = alloca ptr addrspace(4)
  %I.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  store ptr addrspace(4) %G, ptr %G.addr
  store ptr addrspace(4) %L, ptr %L.addr
  store ptr addrspace(4) %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %globalRange1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load ptr addrspace(4), ptr %G.addr
  %localRange = getelementptr inbounds nuw %"group", ptr addrspace(4) %this1, i32 0, i32 1
  %1 = load ptr addrspace(4), ptr %L.addr
  %groupRange = getelementptr inbounds nuw %"group", ptr addrspace(4) %this1, i32 0, i32 2
  %index = getelementptr inbounds nuw %"group", ptr addrspace(4) %this1, i32 0, i32 3
  %2 = load ptr addrspace(4), ptr %I.addr
  ret void
}


define internal spir_func void @InitSize4(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @_ZN7__spirv15getGlobalOffsetILi0EEEmv() 
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZN7__spirv15getGlobalOffsetILi0EEEmv() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z22__spirv_GlobalOffset_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z22__spirv_GlobalOffset_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalOffset, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @InitSize3(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @_ZN7__spirv20getLocalInvocationIdILi0EEEmv() 
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZN7__spirv20getLocalInvocationIdILi0EEEmv() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z27__spirv_LocalInvocationId_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z27__spirv_LocalInvocationId_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @InitSize2(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @_ZN7__spirv14getWorkgroupIdILi0EEEmv() 
  call spir_func void @Foo46(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZN7__spirv14getWorkgroupIdILi0EEEmv() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z21__spirv_WorkgroupId_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z21__spirv_WorkgroupId_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @InitSize1(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @_ZN7__spirv16getNumWorkgroupsILi0EEEmv() 
  call spir_func void @Foo9(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZN7__spirv16getNumWorkgroupsILi0EEEmv() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z23__spirv_NumWorkgroups_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z23__spirv_NumWorkgroups_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @Inv3(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @_ZN7__spirv16getWorkgroupSizeILi0EEEmv() 
  call spir_func void @Foo9(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZN7__spirv16getWorkgroupSizeILi0EEEmv() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z23__spirv_WorkgroupSize_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z23__spirv_WorkgroupSize_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @Foo8(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  %call = call spir_func i64 @_ZN7__spirv13getGlobalSizeILi0EEEmv() 
  call spir_func void @Foo9(ptr addrspace(4) %agg.result, i64 %call) 
  ret void
}


define internal spir_func i64 @_ZN7__spirv13getGlobalSizeILi0EEEmv() {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %call = call spir_func i64 @_Z20__spirv_GlobalSize_xv() 
  ret i64 %call
}


define internal spir_func i64 @_Z20__spirv_GlobalSize_xv()   {
entry:
  %retval = alloca i64
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalSize, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}


define internal spir_func void @Foo30(ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"nd_item", align 1
  store ptr addrspace(4) %f, ptr %f.addr
  %0 = load ptr addrspace(4), ptr %f.addr
  call spir_func void @Foo33(ptr byval(%"nd_item") align 1 %agg.tmp, ptr addrspace(4) %0) 
  ret void
}


define internal spir_func i64 @Foo32(ptr addrspace(4) %this) {
entry:
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %TotalOffset = alloca i64
  %ref.tmp = alloca %class.anon.7
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %TotalOffset.ascast = addrspacecast ptr %TotalOffset to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  store i64 0, ptr %TotalOffset
  %0 = bitcast ptr %ref.tmp to ptr
  store ptr addrspace(4) %this1, ptr %0
  %TotalOffset2 = getelementptr inbounds %class.anon.7, ptr %ref.tmp, i32 0, i32 1
  store ptr addrspace(4) %TotalOffset.ascast, ptr %TotalOffset2
  call spir_func void @Foo34(ptr addrspace(4) %ref.tmp.ascast) 
  %1 = load i64, ptr %TotalOffset
  ret i64 %1
}


define internal spir_func void @Foo34(ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"nd_item", align 1
  store ptr addrspace(4) %f, ptr %f.addr
  %0 = load ptr addrspace(4), ptr %f.addr
  call spir_func void @Foo35(ptr byval(%"nd_item") align 1 %agg.tmp, ptr addrspace(4) %0) 
  ret void
}


define internal spir_func void @Foo35(ptr byval(%"nd_item") align 1 %0, ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"nd_item", align 1
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store ptr addrspace(4) %f, ptr %f.addr
  %2 = load ptr addrspace(4), ptr %f.addr
  %call = call spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %ref.tmp.ascast) 
  call spir_func void @Foo36(ptr addrspace(4) %2, i64 %call) 
  ret void
}


define internal spir_func void @Foo36(ptr addrspace(4) %this, i64 %I)  align 2  {
entry:
  %this.addr = alloca ptr addrspace(4)
  %I.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load ptr addrspace(4), ptr addrspace(4) %0
  %TotalOffset = getelementptr inbounds nuw %class.anon.7, ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load ptr addrspace(4), ptr addrspace(4) %TotalOffset
  %3 = load i64, ptr addrspace(4) %2
  %impl1 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 2
  %4 = load i64, ptr %I.addr
  %conv = trunc i64 %4 to i32
  %call = call spir_func i64 @Foo37(ptr addrspace(4) %MemRange, i32 %conv) 
  %mul = mul i64 %3, %call
  %TotalOffset2 = getelementptr inbounds nuw %class.anon.7, ptr addrspace(4) %this1, i32 0, i32 1
  %5 = load ptr addrspace(4), ptr addrspace(4) %TotalOffset2
  store i64 %mul, ptr addrspace(4) %5
  %impl32 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %Offset3 = bitcast ptr addrspace(4) %impl32 to ptr addrspace(4)
  %6 = load i64, ptr %I.addr
  %conv4 = trunc i64 %6 to i32
  %call5 = call spir_func i64 @Foo37(ptr addrspace(4) %Offset3, i32 %conv4) 
  %TotalOffset6 = getelementptr inbounds nuw %class.anon.7, ptr addrspace(4) %this1, i32 0, i32 1
  %7 = load ptr addrspace(4), ptr addrspace(4) %TotalOffset6
  %8 = load i64, ptr addrspace(4) %7
  %add = add i64 %8, %call5
  store i64 %add, ptr addrspace(4) %7
  ret void
}


define internal spir_func void @Foo33(ptr byval(%"nd_item") align 1 %0, ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"nd_item", align 1
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store ptr addrspace(4) %f, ptr %f.addr
  %2 = load ptr addrspace(4), ptr %f.addr
  %call = call spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %ref.tmp.ascast) 
  call spir_func void @Foo38(ptr addrspace(4) %2, i64 %call) 
  ret void
}


define internal spir_func void @Foo38(ptr addrspace(4) %this, i64 %I)  align 2  {
entry:
  %this.addr = alloca ptr addrspace(4)
  %I.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load ptr addrspace(4), ptr addrspace(4) %0
  %Offset = getelementptr inbounds nuw %class.anon.6, ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load ptr addrspace(4), ptr addrspace(4) %Offset
  %3 = load i64, ptr %I.addr
  %conv = trunc i64 %3 to i32
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %2, i32 %conv) 
  %4 = load i64, ptr addrspace(4) %call
  %call2 = call spir_func ptr addrspace(4) @Foo39(ptr addrspace(4) %1) 
  %5 = load i64, ptr %I.addr
  %conv3 = trunc i64 %5 to i32
  %call4 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %call2, i32 %conv3) 
  store i64 %4, ptr addrspace(4) %call4
  %AccessRange = getelementptr inbounds nuw %class.anon.6, ptr addrspace(4) %this1, i32 0, i32 2
  %6 = load ptr addrspace(4), ptr addrspace(4) %AccessRange
  %7 = load i64, ptr %I.addr
  %conv5 = trunc i64 %7 to i32
  %call6 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %6, i32 %conv5) 
  %8 = load i64, ptr addrspace(4) %call6
  %call7 = call spir_func ptr addrspace(4) @Foo40A(ptr addrspace(4) %1) 
  %9 = load i64, ptr %I.addr
  %conv8 = trunc i64 %9 to i32
  %call9 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %call7, i32 %conv8) 
  store i64 %8, ptr addrspace(4) %call9
  %MemRange = getelementptr inbounds nuw %class.anon.6, ptr addrspace(4) %this1, i32 0, i32 3
  %10 = load ptr addrspace(4), ptr addrspace(4) %MemRange
  %11 = load i64, ptr %I.addr
  %conv10 = trunc i64 %11 to i32
  %call11 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %10, i32 %conv10) 
  %12 = load i64, ptr addrspace(4) %call11
  %call12 = call spir_func ptr addrspace(4) @Foo41A(ptr addrspace(4) %1) 
  %13 = load i64, ptr %I.addr
  %conv13 = trunc i64 %13 to i32
  %call14 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %call12, i32 %conv13) 
  store i64 %12, ptr addrspace(4) %call14
  ret void
}


define internal spir_func ptr addrspace(4) @Foo39(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %Offset2 = bitcast ptr addrspace(4) %impl1 to ptr addrspace(4)
  ret ptr addrspace(4) %Offset2
}


define internal spir_func ptr addrspace(4) @Foo40A(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %AccessRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 1
  ret ptr addrspace(4) %AccessRange
}


define internal spir_func ptr addrspace(4) @Foo41A(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 2
  ret ptr addrspace(4) %MemRange
}


define internal spir_func void @Foo13(ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"nd_item", align 1
  store ptr addrspace(4) %f, ptr %f.addr
  %0 = load ptr addrspace(4), ptr %f.addr
  call spir_func void @Foo14(ptr byval(%"nd_item") align 1 %agg.tmp, ptr addrspace(4) %0) 
  ret void
}


define internal spir_func i64 @Foo21(ptr addrspace(4) %this) {
entry:
  %retval = alloca i64
  %this.addr = alloca ptr addrspace(4)
  %TotalOffset = alloca i64
  %ref.tmp = alloca %class.anon.7
  %cleanup.dest.slot = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %TotalOffset.ascast = addrspacecast ptr %TotalOffset to ptr addrspace(4)
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  store i64 0, ptr %TotalOffset
  %0 = bitcast ptr %ref.tmp to ptr
  store ptr addrspace(4) %this1, ptr %0
  %TotalOffset2 = getelementptr inbounds %class.anon.7, ptr %ref.tmp, i32 0, i32 1
  store ptr addrspace(4) %TotalOffset.ascast, ptr %TotalOffset2
  call spir_func void @Bar83(ptr addrspace(4) %ref.tmp.ascast) 
  %1 = load i64, ptr %TotalOffset
  ret i64 %1
}


define internal spir_func void @Bar83(ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %agg.tmp = alloca %"nd_item", align 1
  store ptr addrspace(4) %f, ptr %f.addr
  %0 = load ptr addrspace(4), ptr %f.addr
  call spir_func void @Bar84(ptr byval(%"nd_item") align 1 %agg.tmp, ptr addrspace(4) %0) 
  ret void
}


define internal spir_func void @Bar84(ptr byval(%"nd_item") align 1 %0, ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"nd_item", align 1
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store ptr addrspace(4) %f, ptr %f.addr
  %2 = load ptr addrspace(4), ptr %f.addr
  %call = call spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %ref.tmp.ascast) 
  call spir_func void @Bar85(ptr addrspace(4) %2, i64 %call) 
  ret void
}


define internal spir_func void @Bar85(ptr addrspace(4) %this, i64 %I)  align 2  {
entry:
  %this.addr = alloca ptr addrspace(4)
  %I.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load ptr addrspace(4), ptr addrspace(4) %0
  %TotalOffset = getelementptr inbounds nuw %class.anon.7, ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load ptr addrspace(4), ptr addrspace(4) %TotalOffset
  %3 = load i64, ptr addrspace(4) %2
  %impl1 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 2
  %4 = load i64, ptr %I.addr
  %conv = trunc i64 %4 to i32
  %call = call spir_func i64 @Foo37(ptr addrspace(4) %MemRange, i32 %conv) 
  %mul = mul i64 %3, %call
  %TotalOffset2 = getelementptr inbounds nuw %class.anon.7, ptr addrspace(4) %this1, i32 0, i32 1
  %5 = load ptr addrspace(4), ptr addrspace(4) %TotalOffset2
  store i64 %mul, ptr addrspace(4) %5
  %impl32 = bitcast ptr addrspace(4) %1 to ptr addrspace(4)
  %Offset3 = bitcast ptr addrspace(4) %impl32 to ptr addrspace(4)
  %6 = load i64, ptr %I.addr
  %conv4 = trunc i64 %6 to i32
  %call5 = call spir_func i64 @Foo37(ptr addrspace(4) %Offset3, i32 %conv4) 
  %TotalOffset6 = getelementptr inbounds nuw %class.anon.7, ptr addrspace(4) %this1, i32 0, i32 1
  %7 = load ptr addrspace(4), ptr addrspace(4) %TotalOffset6
  %8 = load i64, ptr addrspace(4) %7
  %add = add i64 %8, %call5
  store i64 %add, ptr addrspace(4) %7
  ret void
}


define internal spir_func void @Foo14(ptr byval(%"nd_item") align 1 %0, ptr addrspace(4) %f) {
entry:
  %f.addr = alloca ptr addrspace(4)
  %ref.tmp = alloca %"nd_item", align 1
  %ref.tmp.ascast = addrspacecast ptr %ref.tmp to ptr addrspace(4)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  store ptr addrspace(4) %f, ptr %f.addr
  %2 = load ptr addrspace(4), ptr %f.addr
  %call = call spir_func i64 @_ZNKSt17integral_constantImLm0EEcvmEv(ptr addrspace(4) align 1 %ref.tmp.ascast) 
  call spir_func void @Foo15(ptr addrspace(4) %2, i64 %call) 
  ret void
}


define internal spir_func void @Foo15(ptr addrspace(4) %this, i64 %I)  align 2  {
entry:
  %this.addr = alloca ptr addrspace(4)
  %I.addr = alloca i64
  store ptr addrspace(4) %this, ptr %this.addr
  store i64 %I, ptr %I.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load ptr addrspace(4), ptr addrspace(4) %0
  %Offset = getelementptr inbounds nuw %class.anon.6, ptr addrspace(4) %this1, i32 0, i32 1
  %2 = load ptr addrspace(4), ptr addrspace(4) %Offset
  %3 = load i64, ptr %I.addr
  %conv = trunc i64 %3 to i32
  %call = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %2, i32 %conv) 
  %4 = load i64, ptr addrspace(4) %call
  %call2 = call spir_func ptr addrspace(4) @Foo17(ptr addrspace(4) %1) 
  %5 = load i64, ptr %I.addr
  %conv3 = trunc i64 %5 to i32
  %call4 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %call2, i32 %conv3) 
  store i64 %4, ptr addrspace(4) %call4
  %AccessRange = getelementptr inbounds nuw %class.anon.6, ptr addrspace(4) %this1, i32 0, i32 2
  %6 = load ptr addrspace(4), ptr addrspace(4) %AccessRange
  %7 = load i64, ptr %I.addr
  %conv5 = trunc i64 %7 to i32
  %call6 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %6, i32 %conv5) 
  %8 = load i64, ptr addrspace(4) %call6
  %call7 = call spir_func ptr addrspace(4) @Foo18(ptr addrspace(4) %1) 
  %9 = load i64, ptr %I.addr
  %conv8 = trunc i64 %9 to i32
  %call9 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %call7, i32 %conv8) 
  store i64 %8, ptr addrspace(4) %call9
  %MemRange = getelementptr inbounds nuw %class.anon.6, ptr addrspace(4) %this1, i32 0, i32 3
  %10 = load ptr addrspace(4), ptr addrspace(4) %MemRange
  %11 = load i64, ptr %I.addr
  %conv10 = trunc i64 %11 to i32
  %call11 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %10, i32 %conv10) 
  %12 = load i64, ptr addrspace(4) %call11
  %call12 = call spir_func ptr addrspace(4) @Foo19(ptr addrspace(4) %1) 
  %13 = load i64, ptr %I.addr
  %conv13 = trunc i64 %13 to i32
  %call14 = call spir_func ptr addrspace(4) @Foo16(ptr addrspace(4) %call12, i32 %conv13) 
  store i64 %12, ptr addrspace(4) %call14
  ret void
}


define internal spir_func ptr addrspace(4) @Foo17(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %Offset2 = bitcast ptr addrspace(4) %impl1 to ptr addrspace(4)
  ret ptr addrspace(4) %Offset2
}


define internal spir_func ptr addrspace(4) @Foo18(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %AccessRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 1
  ret ptr addrspace(4) %AccessRange
}


define internal spir_func ptr addrspace(4) @Foo19(ptr addrspace(4) %this) {
entry:
  %retval = alloca ptr addrspace(4)
  %this.addr = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %impl1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %impl1, i32 0, i32 2
  ret ptr addrspace(4) %MemRange
}


declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) 


define internal spir_func void @Foo12(ptr addrspace(4) dead_on_unwind noalias writable sret(%"range") %agg.result) {
entry:
  call spir_func void @Foo9(ptr addrspace(4) %agg.result, i64 0) 
  ret void
}


define internal spir_func void @Foo10(ptr addrspace(4) %this, ptr byval(%"range") %Offset, ptr byval(%"range") %AccessRange, ptr byval(%"range") %MemoryRange) unnamed_addr {
entry:
  %this.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr
  %this1 = load ptr addrspace(4), ptr %this.addr
  %Offset21 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %AccessRange3 = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %this1, i32 0, i32 1
  %MemRange = getelementptr inbounds nuw %"detail::AccessorImplDevice", ptr addrspace(4) %this1, i32 0, i32 2
  ret void
}


define internal spir_func void @__assert_fail(ptr addrspace(4) %expr, ptr addrspace(4) %file, i32 %line, ptr addrspace(4) %func)   {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() 
  %call1 = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() 
  %call2 = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_zv() 
  %call3 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_xv() 
  %call4 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_yv() 
  %call5 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_zv() 
  tail call spir_func void @__devicelib_assert_fail(ptr addrspace(4) %expr, ptr addrspace(4) %file, i32 %line, ptr addrspace(4) %func, i64 %call, i64 %call1, i64 %call2, i64 %call3, i64 %call4, i64 %call5) 
  ret void
}


define internal spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() local_unnamed_addr   {
entry:
  %0 = getelementptr inbounds i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8
  %1 = load i64, ptr addrspace(1) %0
  ret i64 %1
}


define internal spir_func i64 @_Z28__spirv_GlobalInvocationId_zv() local_unnamed_addr   {
entry:
  %0 = getelementptr inbounds i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 16
  %1 = load i64, ptr addrspace(1) %0, align 16
  ret i64 %1
}


define internal spir_func i64 @_Z27__spirv_LocalInvocationId_yv() local_unnamed_addr   {
entry:
  %0 = getelementptr inbounds i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8
  %1 = load i64, ptr addrspace(1) %0
  ret i64 %1
}


define internal spir_func i64 @_Z27__spirv_LocalInvocationId_zv() local_unnamed_addr   {
entry:
  %0 = getelementptr inbounds i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 16
  %1 = load i64, ptr addrspace(1) %0, align 16
  ret i64 %1
}


define internal spir_func void @__devicelib_assert_fail(ptr addrspace(4) %expr, ptr addrspace(4) %file, i32 %line, ptr addrspace(4) %func, i64 %gid0, i64 %gid1, i64 %gid2, i64 %lid0, i64 %lid1, i64 %lid2) local_unnamed_addr   {
entry:
  %call.i = tail call spir_func i32 @_Z29__spirv_AtomicCompareExchangePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ii(ptr addrspace(1) @SPIR_AssertHappenedMem, i32 1, i32 16, i32 16, i32 1, i32 0) 
  %cmp = icmp eq i32 %call.i, 0
  %0 = getelementptr inbounds nuw i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 4
  %1 = getelementptr inbounds nuw i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 261
  %2 = getelementptr inbounds nuw i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 518
  br i1 %cmp, label %if.then, label %if.end82

if.then:                                          ; preds = %entry
  store i32 %line, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 648)
  store i64 %gid0, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 656)
  store i64 %gid1, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 664)
  store i64 %gid2, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 672)
  store i64 %lid0, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 680)
  store i64 %lid1, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 688)
  store i64 %lid2, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @SPIR_AssertHappenedMem, i64 696)
  %tobool.not = icmp eq ptr addrspace(4) %expr, null
  br i1 %tobool.not, label %if.end, label %for.cond.preheader

for.cond.preheader:                               ; preds = %if.then
  br label %for.cond

for.cond:                                         ; preds = %for.cond.preheader, %for.inc
  %ExprLength.0 = phi i32 [ %inc, %for.inc ], [ 0, %for.cond.preheader ]
  %C.0 = phi ptr addrspace(4) [ %incdec.ptr, %for.inc ], [ %expr, %for.cond.preheader ]
  %3 = load i8, ptr addrspace(4) %C.0, align 1
  %cmp2.not = icmp eq i8 %3, 0
  br i1 %cmp2.not, label %if.end, label %for.inc

for.inc:                                          ; preds = %for.cond
  %incdec.ptr = getelementptr inbounds nuw i8, ptr addrspace(4) %C.0, i64 1
  %inc = add nuw nsw i32 %ExprLength.0, 1
  br label %for.cond

if.end:                                           ; preds = %for.cond, %if.then
  %ExprLength.1 = phi i32 [ 0, %if.then ], [ %ExprLength.0, %for.cond ]
  %tobool3.not = icmp eq ptr addrspace(4) %file, null
  br i1 %tobool3.not, label %if.end16, label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %if.end
  br label %for.cond6

for.cond6:                                        ; preds = %for.cond6.preheader, %for.inc12
  %FileLength.0 = phi i32 [ %inc14, %for.inc12 ], [ 0, %for.cond6.preheader ]
  %C5.0 = phi ptr addrspace(4) [ %incdec.ptr13, %for.inc12 ], [ %file, %for.cond6.preheader ]
  %4 = load i8, ptr addrspace(4) %C5.0, align 1
  %cmp8.not = icmp eq i8 %4, 0
  br i1 %cmp8.not, label %if.end16, label %for.inc12

for.inc12:                                        ; preds = %for.cond6
  %incdec.ptr13 = getelementptr inbounds nuw i8, ptr addrspace(4) %C5.0, i64 1
  %inc14 = add nuw nsw i32 %FileLength.0, 1
  br label %for.cond6

if.end16:                                         ; preds = %for.cond6, %if.end
  %FileLength.1 = phi i32 [ 0, %if.end ], [ %FileLength.0, %for.cond6 ]
  %tobool17.not = icmp eq ptr addrspace(4) %func, null
  br i1 %tobool17.not, label %if.end30.thread, label %for.cond20.preheader

for.cond20.preheader:                             ; preds = %if.end16
  br label %for.cond20

for.cond20:                                       ; preds = %for.cond20.preheader, %for.inc26
  %FuncLength.0 = phi i32 [ %inc28, %for.inc26 ], [ 0, %for.cond20.preheader ]
  %C19.0 = phi ptr addrspace(4) [ %incdec.ptr27, %for.inc26 ], [ %func, %for.cond20.preheader ]
  %5 = load i8, ptr addrspace(4) %C19.0, align 1
  %cmp22.not = icmp eq i8 %5, 0
  br i1 %cmp22.not, label %if.end30, label %for.inc26

for.inc26:                                        ; preds = %for.cond20
  %incdec.ptr27 = getelementptr inbounds nuw i8, ptr addrspace(4) %C19.0, i64 1
  %inc28 = add i32 %FuncLength.0, 1
  br label %for.cond20

if.end30:                                         ; preds = %for.cond20
  %spec.select = tail call i32 @llvm.umin.i32(i32 %ExprLength.1, i32 256)
  %MaxFileIdx.0 = tail call i32 @llvm.umin.i32(i32 %FileLength.1, i32 256)
  %spec.select126 = tail call i32 @llvm.umin.i32(i32 %FuncLength.0, i32 128)
  br label %6

if.end30.thread:                                  ; preds = %if.end16
  %spec.select116 = tail call i32 @llvm.umin.i32(i32 %ExprLength.1, i32 256)
  %MaxFileIdx.0118 = tail call i32 @llvm.umin.i32(i32 %FileLength.1, i32 256)
  br label %6

6:                                                ; preds = %if.end30, %if.end30.thread
  %MaxFileIdx.0124 = phi i32 [ %MaxFileIdx.0118, %if.end30.thread ], [ %MaxFileIdx.0, %if.end30 ]
  %spec.select122 = phi i32 [ %spec.select116, %if.end30.thread ], [ %spec.select, %if.end30 ]
  %7 = phi i32 [ 0, %if.end30.thread ], [ %spec.select126, %if.end30 ]
  br label %for.cond40

for.cond40:                                       ; preds = %for.body44, %6
  %lsr.iv9 = phi ptr addrspace(4) [ %scevgep10, %for.body44 ], [ %expr, %6 ]
  %lsr.iv7 = phi ptr addrspace(1) [ %scevgep8, %for.body44 ], [ %0, %6 ]
  %Idx.0 = phi i32 [ 0, %6 ], [ %inc48, %for.body44 ]
  %cmp41 = icmp ult i32 %Idx.0, %spec.select122
  br i1 %cmp41, label %for.body44, label %for.cond.cleanup42

for.cond.cleanup42:                               ; preds = %for.cond40
  %idxprom50 = zext nneg i32 %spec.select122 to i64
  %arrayidx51 = getelementptr inbounds [257 x i8], ptr addrspace(1) %0, i64 0, i64 %idxprom50
  store i8 0, ptr addrspace(1) %arrayidx51, align 1
  br label %for.cond53

for.cond53:                                       ; preds = %for.body57, %for.cond.cleanup42
  %lsr.iv5 = phi ptr addrspace(4) [ %scevgep6, %for.body57 ], [ %file, %for.cond.cleanup42 ]
  %lsr.iv3 = phi ptr addrspace(1) [ %scevgep4, %for.body57 ], [ %1, %for.cond.cleanup42 ]
  %Idx52.0 = phi i32 [ 0, %for.cond.cleanup42 ], [ %inc63, %for.body57 ]
  %cmp54 = icmp ult i32 %Idx52.0, %MaxFileIdx.0124
  br i1 %cmp54, label %for.body57, label %for.cond.cleanup55

for.cond.cleanup55:                               ; preds = %for.cond53
  %idxprom65 = zext nneg i32 %MaxFileIdx.0124 to i64
  %arrayidx66 = getelementptr inbounds [257 x i8], ptr addrspace(1) %1, i64 0, i64 %idxprom65
  store i8 0, ptr addrspace(1) %arrayidx66, align 1
  br label %for.cond68

for.cond68:                                       ; preds = %for.body72, %for.cond.cleanup55
  %lsr.iv1 = phi ptr addrspace(4) [ %scevgep2, %for.body72 ], [ %func, %for.cond.cleanup55 ]
  %lsr.iv = phi ptr addrspace(1) [ %scevgep, %for.body72 ], [ %2, %for.cond.cleanup55 ]
  %Idx67.0 = phi i32 [ 0, %for.cond.cleanup55 ], [ %inc78, %for.body72 ]
  %cmp69 = icmp ult i32 %Idx67.0, %7
  br i1 %cmp69, label %for.body72, label %for.cond.cleanup70

for.cond.cleanup70:                               ; preds = %for.cond68
  %idxprom80 = zext nneg i32 %7 to i64
  %arrayidx81 = getelementptr inbounds [129 x i8], ptr addrspace(1) %2, i64 0, i64 %idxprom80
  store i8 0, ptr addrspace(1) %arrayidx81, align 1
  tail call spir_func void @_Z19__spirv_AtomicStorePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEi(ptr addrspace(1) @SPIR_AssertHappenedMem, i32 1, i32 16, i32 2) 
  br label %if.end82

if.end82:                                         ; preds = %for.cond.cleanup70, %entry
  ret void

for.body72:                                       ; preds = %for.cond68
  %8 = load i8, ptr addrspace(4) %lsr.iv1, align 1
  store i8 %8, ptr addrspace(1) %lsr.iv, align 1
  %inc78 = add nuw nsw i32 %Idx67.0, 1
  %scevgep = getelementptr i8, ptr addrspace(1) %lsr.iv, i64 1
  %scevgep2 = getelementptr i8, ptr addrspace(4) %lsr.iv1, i64 1
  br label %for.cond68

for.body57:                                       ; preds = %for.cond53
  %9 = load i8, ptr addrspace(4) %lsr.iv5, align 1
  store i8 %9, ptr addrspace(1) %lsr.iv3, align 1
  %inc63 = add nuw nsw i32 %Idx52.0, 1
  %scevgep4 = getelementptr i8, ptr addrspace(1) %lsr.iv3, i64 1
  %scevgep6 = getelementptr i8, ptr addrspace(4) %lsr.iv5, i64 1
  br label %for.cond53

for.body44:                                       ; preds = %for.cond40
  %10 = load i8, ptr addrspace(4) %lsr.iv9, align 1
  store i8 %10, ptr addrspace(1) %lsr.iv7, align 1
  %inc48 = add nuw nsw i32 %Idx.0, 1
  %scevgep8 = getelementptr i8, ptr addrspace(1) %lsr.iv7, i64 1
  %scevgep10 = getelementptr i8, ptr addrspace(4) %lsr.iv9, i64 1
  br label %for.cond40
}

declare extern_weak dso_local spir_func i32 @_Z29__spirv_AtomicCompareExchangePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ii(ptr addrspace(1), i32, i32, i32, i32, i32) local_unnamed_addr 
declare extern_weak dso_local spir_func void @_Z19__spirv_AtomicStorePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEi(ptr addrspace(1), i32, i32, i32) local_unnamed_addr 
declare i32 @llvm.umin.i32(i32, i32) 
