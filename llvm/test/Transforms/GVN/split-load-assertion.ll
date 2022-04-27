; RUN: opt -S -O3 < %s

; This test had caused a GVN assertion error in an earlier version

target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-img-unknown-elf"

%"class.std::function" = type { %"class.std::_Function_base", ptr }
%"class.std::_Function_base" = type { %"union.std::_Any_data", ptr }
%"union.std::_Any_data" = type { %"union.std::_Nocopy_types" }
%"union.std::_Nocopy_types" = type { { i32, i32 } }

; Function Attrs: mustprogress
define dso_local noundef i32 @_Z3foov() local_unnamed_addr personality ptr @__gxx_personality_v0 {
entry:
  %f = alloca %"class.std::function", align 4
  %g = alloca %"class.std::function", align 4
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %f) 
  call void @_ZNSt8functionIFivEEC2IRS0_vEEOT_(ptr noundef nonnull align 4 dereferenceable(16) %f, ptr noundef nonnull @_Z2f1v) 
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %g) 
  call void @_ZNSt8functionIFivEEC2IRS0_vEEOT_(ptr noundef nonnull align 4 dereferenceable(16) %g, ptr noundef nonnull @_Z2f2v) 
  call void @_ZNSt8functionIFivEE4swapERS1_(ptr noundef nonnull align 4 dereferenceable(16) %f, ptr noundef nonnull align 4 dereferenceable(16) %g) 
  %call = invoke noundef i32 @_ZNKSt8functionIFivEEclEv(ptr noundef nonnull align 4 dereferenceable(16) %g)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %cmp = icmp eq i32 %call, 42
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %invoke.cont
  %call2 = invoke noundef i32 @_ZNKSt8functionIFivEEclEv(ptr noundef nonnull align 4 dereferenceable(16) %f)
          to label %invoke.cont1 unwind label %lpad

invoke.cont1:                                     ; preds = %land.rhs
  %cmp3 = icmp eq i32 %call2, 0
  %phi.cast = zext i1 %cmp3 to i32
  br label %land.end

land.end:                                         ; preds = %invoke.cont1, %invoke.cont
  %0 = phi i32 [ 0, %invoke.cont ], [ %phi.cast, %invoke.cont1 ]
  call void @_ZNSt14_Function_baseD2Ev(ptr noundef nonnull align 4 dereferenceable(16) %g) 
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %g) 
  call void @_ZNSt14_Function_baseD2Ev(ptr noundef nonnull align 4 dereferenceable(16) %f) 
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %f) 
  ret i32 %0

lpad:                                             ; preds = %land.rhs, %entry
  %1 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt14_Function_baseD2Ev(ptr noundef nonnull align 4 dereferenceable(16) %g) 
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %g) 
  call void @_ZNSt14_Function_baseD2Ev(ptr noundef nonnull align 4 dereferenceable(16) %f) 
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %f) 
  resume { ptr, i32 } %1
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg %0, ptr nocapture %1) 

declare dso_local noundef i32 @_Z2f1v() 

; Function Attrs: nounwind
define linkonce_odr dso_local void @_ZNSt8functionIFivEEC2IRS0_vEEOT_(ptr noundef nonnull align 4 dereferenceable(16) %this, ptr noundef nonnull %__f) unnamed_addr align 2 {
entry:
  call void @llvm.memset.p0.i32(ptr noundef nonnull align 4 dereferenceable(12) %this, i8 0, i32 12, i1 false)
  call void @_ZNSt14_Function_baseC2Ev(ptr noundef nonnull align 4 dereferenceable(12) %this) 
  %_M_invoker = getelementptr inbounds %"class.std::function", ptr %this, i32 0, i32 1
  store ptr null, ptr %_M_invoker, align 4
  %call = call noundef zeroext i1 @_ZNSt14_Function_base13_Base_managerIPFivEE21_M_not_empty_functionIS1_EEbPT_(ptr noundef nonnull %__f) 
  br i1 %call, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @_ZNSt14_Function_base13_Base_managerIPFivEE15_M_init_functorIRS1_EEvRSt9_Any_dataOT_(ptr noundef nonnull align 4 dereferenceable(8) %this, ptr noundef nonnull %__f) 
  store ptr @_ZNSt17_Function_handlerIFivEPS0_E9_M_invokeERKSt9_Any_data, ptr %_M_invoker, align 4
  %_M_manager = getelementptr inbounds %"class.std::_Function_base", ptr %this, i32 0, i32 1
  store ptr @_ZNSt17_Function_handlerIFivEPS0_E10_M_managerERSt9_Any_dataRKS3_St18_Manager_operation, ptr %_M_manager, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare dso_local noundef i32 @_Z2f2v() 

; Function Attrs: mustprogress nounwind
define linkonce_odr dso_local void @_ZNSt8functionIFivEE4swapERS1_(ptr noundef nonnull align 4 dereferenceable(16) %this, ptr noundef nonnull align 4 dereferenceable(16) %__x) local_unnamed_addr align 2 {
entry:
  call void @_ZSt4swapISt9_Any_dataENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SD_(ptr noundef nonnull align 4 dereferenceable(8) %this, ptr noundef nonnull align 4 dereferenceable(8) %__x) 
  %_M_manager = getelementptr inbounds %"class.std::_Function_base", ptr %this, i32 0, i32 1
  %_M_manager3 = getelementptr inbounds %"class.std::_Function_base", ptr %__x, i32 0, i32 1
  call void @_ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SJ_(ptr noundef nonnull align 4 dereferenceable(4) %_M_manager, ptr noundef nonnull align 4 dereferenceable(4) %_M_manager3) 
  %_M_invoker = getelementptr inbounds %"class.std::function", ptr %this, i32 0, i32 1
  %_M_invoker4 = getelementptr inbounds %"class.std::function", ptr %__x, i32 0, i32 1
  call void @_ZSt4swapIPFiRKSt9_Any_dataEENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS8_ESt18is_move_assignableIS8_EEE5valueEvE4typeERS8_SH_(ptr noundef nonnull align 4 dereferenceable(4) %_M_invoker, ptr noundef nonnull align 4 dereferenceable(4) %_M_invoker4) 
  ret void
}

; Function Attrs: mustprogress
declare dso_local noundef i32 @_ZNKSt8functionIFivEEclEv(ptr noundef nonnull align 4 dereferenceable(16) %this) local_unnamed_addr align 2 

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare dso_local void @_ZNSt14_Function_baseD2Ev(ptr noundef nonnull align 4 dereferenceable(12) %this) unnamed_addr align 2 

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg %0, ptr nocapture %1) 

; Function Attrs: argmemonly mustprogress nocallback nofree nounwind willreturn writeonly
declare void @llvm.memset.p0.i32(ptr nocapture writeonly %0, i8 %1, i32 %2, i1 immarg %3) 

; Function Attrs: nounwind
define linkonce_odr dso_local void @_ZNSt14_Function_baseC2Ev(ptr noundef nonnull align 4 dereferenceable(12) %this) unnamed_addr align 2 {
entry:
  store ptr null, ptr %this, align 4
  %_M_manager = getelementptr inbounds %"class.std::_Function_base", ptr %this, i32 0, i32 1
  store ptr null, ptr %_M_manager, align 4
  ret void
}

; Function Attrs: mustprogress nounwind
declare dso_local noundef zeroext i1 @_ZNSt14_Function_base13_Base_managerIPFivEE21_M_not_empty_functionIS1_EEbPT_(ptr noundef %__fp) local_unnamed_addr align 2 

; Function Attrs: mustprogress nounwind
define linkonce_odr dso_local void @_ZNSt14_Function_base13_Base_managerIPFivEE15_M_init_functorIRS1_EEvRSt9_Any_dataOT_(ptr noundef nonnull align 4 dereferenceable(8) %__functor, ptr noundef nonnull %__f) local_unnamed_addr align 2 personality ptr @__gxx_personality_v0 {
entry:
  call void @_ZNSt14_Function_base13_Base_managerIPFivEE9_M_createIRS1_EEvRSt9_Any_dataOT_St17integral_constantIbLb1EE(ptr noundef nonnull align 4 dereferenceable(8) %__functor, ptr noundef nonnull %__f)
  ret void
}

; Function Attrs: mustprogress
declare dso_local noundef i32 @_ZNSt17_Function_handlerIFivEPS0_E9_M_invokeERKSt9_Any_data(ptr noundef nonnull align 4 dereferenceable(8) %__functor) align 2 

; Function Attrs: mustprogress
declare dso_local noundef zeroext i1 @_ZNSt17_Function_handlerIFivEPS0_E10_M_managerERSt9_Any_dataRKS3_St18_Manager_operation(ptr noundef nonnull align 4 dereferenceable(8) %__dest, ptr noundef nonnull align 4 dereferenceable(8) %__source, i32 noundef %__op) align 2 

; Function Attrs: mustprogress nounwind
define linkonce_odr dso_local void @_ZNSt14_Function_base13_Base_managerIPFivEE9_M_createIRS1_EEvRSt9_Any_dataOT_St17integral_constantIbLb1EE(ptr noundef nonnull align 4 dereferenceable(8) %__dest, ptr noundef nonnull %__f) local_unnamed_addr align 2 {
entry:
  %call = call noundef ptr @_ZNSt9_Any_data9_M_accessEv(ptr noundef nonnull align 4 dereferenceable(8) %__dest) 
  store ptr %__f, ptr %call, align 4
  ret void
}

; Function Attrs: mustprogress nounwind
define linkonce_odr dso_local noundef ptr @_ZNSt9_Any_data9_M_accessEv(ptr noundef nonnull align 4 dereferenceable(8) %this) local_unnamed_addr align 2 {
entry:
  ret ptr %this
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr dso_local void @_ZSt4swapISt9_Any_dataENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SD_(ptr noundef nonnull align 4 dereferenceable(8) %__a, ptr noundef nonnull align 4 dereferenceable(8) %__b) local_unnamed_addr {
entry:
  %0 = load i64, ptr %__a, align 4
  %1 = load i64, ptr %__b, align 4
  store i64 %1, ptr %__a, align 4
  store i64 %0, ptr %__b, align 4
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind
declare dso_local void @_ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SJ_(ptr noundef nonnull align 4 dereferenceable(4) %__a, ptr noundef nonnull align 4 dereferenceable(4) %__b) local_unnamed_addr 

; Function Attrs: inlinehint mustprogress nounwind
declare dso_local void @_ZSt4swapIPFiRKSt9_Any_dataEENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS8_ESt18is_move_assignableIS8_EEE5valueEvE4typeERS8_SH_(ptr noundef nonnull align 4 dereferenceable(4) %__a, ptr noundef nonnull align 4 dereferenceable(4) %__b) local_unnamed_addr 

