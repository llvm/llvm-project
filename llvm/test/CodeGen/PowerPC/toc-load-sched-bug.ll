; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; This test checks for misordering of a TOC restore instruction relative
; to subsequent uses of the TOC register.  Previously this test broke
; because there was no TOC register dependency between the instructions,
; and the usual stack-adjust instructions that held the TOC restore in
; place were optimized away.

%"class.llvm::Module" = type { %"class.llvm::LLVMContext"*, %"class.llvm::iplist", %"class.llvm::iplist.0", %"class.llvm::iplist.9", %"struct.llvm::ilist", %"class.std::basic_string", %"class.llvm::ValueSymbolTable"*, %"class.llvm::StringMap", %"class.std::unique_ptr", %"class.std::basic_string", %"class.std::basic_string", i8*, %"class.llvm::RandomNumberGenerator"*, %"class.std::basic_string", %"class.llvm::DataLayout" }
%"class.llvm::iplist" = type { %"struct.llvm::ilist_traits", %"class.llvm::GlobalVariable"* }
%"struct.llvm::ilist_traits" = type { %"class.llvm::ilist_node" }
%"class.llvm::ilist_node" = type { %"class.llvm::ilist_half_node", %"class.llvm::GlobalVariable"* }
%"class.llvm::ilist_half_node" = type { %"class.llvm::GlobalVariable"* }
%"class.llvm::GlobalVariable" = type { %"class.llvm::GlobalObject", %"class.llvm::ilist_node", i8 }
%"class.llvm::GlobalObject" = type { %"class.llvm::GlobalValue", %"class.std::basic_string", %"class.llvm::Comdat"* }
%"class.llvm::GlobalValue" = type { %"class.llvm::Constant", i32, %"class.llvm::Module"* }
%"class.llvm::Constant" = type { %"class.llvm::User" }
%"class.llvm::User" = type { %"class.llvm::Value.base", i32, %"class.llvm::Use"* }
%"class.llvm::Value.base" = type <{ i32 (...)**, %"class.llvm::Type"*, %"class.llvm::Use"*, %"class.llvm::StringMapEntry"*, i8, i8, i16 }>
%"class.llvm::Type" = type { %"class.llvm::LLVMContext"*, i32, i32, %"class.llvm::Type"** }
%"class.llvm::StringMapEntry" = type opaque
%"class.llvm::Use" = type { %"class.llvm::Value"*, %"class.llvm::Use"*, %"class.llvm::PointerIntPair" }
%"class.llvm::Value" = type { i32 (...)**, %"class.llvm::Type"*, %"class.llvm::Use"*, %"class.llvm::StringMapEntry"*, i8, i8, i16 }
%"class.llvm::PointerIntPair" = type { i64 }
%"class.llvm::Comdat" = type { %"class.llvm::StringMapEntry.43"*, i32 }
%"class.llvm::StringMapEntry.43" = type opaque
%"class.llvm::iplist.0" = type { %"struct.llvm::ilist_traits.1", %"class.llvm::Function"* }
%"struct.llvm::ilist_traits.1" = type { %"class.llvm::ilist_node.7" }
%"class.llvm::ilist_node.7" = type { %"class.llvm::ilist_half_node.8", %"class.llvm::Function"* }
%"class.llvm::ilist_half_node.8" = type { %"class.llvm::Function"* }
%"class.llvm::Function" = type { %"class.llvm::GlobalObject", %"class.llvm::ilist_node.7", %"class.llvm::iplist.44", %"class.llvm::iplist.52", %"class.llvm::ValueSymbolTable"*, %"class.llvm::AttributeSet" }
%"class.llvm::iplist.44" = type { %"struct.llvm::ilist_traits.45", %"class.llvm::BasicBlock"* }
%"struct.llvm::ilist_traits.45" = type { %"class.llvm::ilist_half_node.51" }
%"class.llvm::ilist_half_node.51" = type { %"class.llvm::BasicBlock"* }
%"class.llvm::BasicBlock" = type { %"class.llvm::Value.base", %"class.llvm::ilist_node.61", %"class.llvm::iplist.62", %"class.llvm::Function"* }
%"class.llvm::ilist_node.61" = type { %"class.llvm::ilist_half_node.51", %"class.llvm::BasicBlock"* }
%"class.llvm::iplist.62" = type { %"struct.llvm::ilist_traits.63", %"class.llvm::Instruction"* }
%"struct.llvm::ilist_traits.63" = type { %"class.llvm::ilist_half_node.69" }
%"class.llvm::ilist_half_node.69" = type { %"class.llvm::Instruction"* }
%"class.llvm::Instruction" = type { %"class.llvm::User", %"class.llvm::ilist_node.70", %"class.llvm::BasicBlock"*, %"class.llvm::DebugLoc" }
%"class.llvm::ilist_node.70" = type { %"class.llvm::ilist_half_node.69", %"class.llvm::Instruction"* }
%"class.llvm::DebugLoc" = type { i32, i32 }
%"class.llvm::iplist.52" = type { %"struct.llvm::ilist_traits.53", %"class.llvm::Argument"* }
%"struct.llvm::ilist_traits.53" = type { %"class.llvm::ilist_half_node.59" }
%"class.llvm::ilist_half_node.59" = type { %"class.llvm::Argument"* }
%"class.llvm::Argument" = type { %"class.llvm::Value.base", %"class.llvm::ilist_node.60", %"class.llvm::Function"* }
%"class.llvm::ilist_node.60" = type { %"class.llvm::ilist_half_node.59", %"class.llvm::Argument"* }
%"class.llvm::AttributeSet" = type { %"class.llvm::AttributeSetImpl"* }
%"class.llvm::AttributeSetImpl" = type opaque
%"class.llvm::iplist.9" = type { %"struct.llvm::ilist_traits.10", %"class.llvm::GlobalAlias"* }
%"struct.llvm::ilist_traits.10" = type { %"class.llvm::ilist_node.16" }
%"class.llvm::ilist_node.16" = type { %"class.llvm::ilist_half_node.17", %"class.llvm::GlobalAlias"* }
%"class.llvm::ilist_half_node.17" = type { %"class.llvm::GlobalAlias"* }
%"class.llvm::GlobalAlias" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.16" }
%"struct.llvm::ilist" = type { %"class.llvm::iplist.18" }
%"class.llvm::iplist.18" = type { %"struct.llvm::ilist_traits.19", %"class.llvm::NamedMDNode"* }
%"struct.llvm::ilist_traits.19" = type { %"class.llvm::ilist_node.24" }
%"class.llvm::ilist_node.24" = type { %"class.llvm::ilist_half_node.25", %"class.llvm::NamedMDNode"* }
%"class.llvm::ilist_half_node.25" = type { %"class.llvm::NamedMDNode"* }
%"class.llvm::NamedMDNode" = type { %"class.llvm::ilist_node.24", %"class.std::basic_string", %"class.llvm::Module"*, i8* }
%"class.llvm::ValueSymbolTable" = type opaque
%"class.llvm::StringMap" = type { %"class.llvm::StringMapImpl", %"class.llvm::MallocAllocator" }
%"class.llvm::StringMapImpl" = type { %"class.llvm::StringMapEntryBase"**, i32, i32, i32, i32 }
%"class.llvm::StringMapEntryBase" = type { i32 }
%"class.llvm::MallocAllocator" = type { i8 }
%"class.std::unique_ptr" = type { %"class.std::tuple" }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Head_base.28" }
%"struct.std::_Head_base.28" = type { %"class.llvm::GVMaterializer"* }
%"class.llvm::GVMaterializer" = type opaque
%"class.llvm::RandomNumberGenerator" = type opaque
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%"class.llvm::DataLayout" = type { i8, i32, i32, [4 x i8], %"class.llvm::SmallVector", %"class.llvm::SmallVector.29", %"class.llvm::SmallVector.36", i8* }
%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl.base", %"struct.llvm::SmallVectorStorage" }
%"class.llvm::SmallVectorImpl.base" = type { %"class.llvm::SmallVectorTemplateBase.base" }
%"class.llvm::SmallVectorTemplateBase.base" = type { %"class.llvm::SmallVectorTemplateCommon.base" }
%"class.llvm::SmallVectorTemplateCommon.base" = type <{ %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion" }>
%"class.llvm::SmallVectorBase" = type { i8*, i8*, i8* }
%"struct.llvm::AlignedCharArrayUnion" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::AlignedCharArray" = type { [1 x i8] }
%"struct.llvm::SmallVectorStorage" = type { [7 x %"struct.llvm::AlignedCharArrayUnion"] }
%"class.llvm::SmallVector.29" = type { %"class.llvm::SmallVectorImpl.30", %"struct.llvm::SmallVectorStorage.35" }
%"class.llvm::SmallVectorImpl.30" = type { %"class.llvm::SmallVectorTemplateBase.31" }
%"class.llvm::SmallVectorTemplateBase.31" = type { %"class.llvm::SmallVectorTemplateCommon.32" }
%"class.llvm::SmallVectorTemplateCommon.32" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.33" }
%"struct.llvm::AlignedCharArrayUnion.33" = type { %"struct.llvm::AlignedCharArray.34" }
%"struct.llvm::AlignedCharArray.34" = type { [8 x i8] }
%"struct.llvm::SmallVectorStorage.35" = type { [15 x %"struct.llvm::AlignedCharArrayUnion.33"] }
%"class.llvm::SmallVector.36" = type { %"class.llvm::SmallVectorImpl.37", %"struct.llvm::SmallVectorStorage.42" }
%"class.llvm::SmallVectorImpl.37" = type { %"class.llvm::SmallVectorTemplateBase.38" }
%"class.llvm::SmallVectorTemplateBase.38" = type { %"class.llvm::SmallVectorTemplateCommon.39" }
%"class.llvm::SmallVectorTemplateCommon.39" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.40" }
%"struct.llvm::AlignedCharArrayUnion.40" = type { %"struct.llvm::AlignedCharArray.41" }
%"struct.llvm::AlignedCharArray.41" = type { [16 x i8] }
%"struct.llvm::SmallVectorStorage.42" = type { [7 x %"struct.llvm::AlignedCharArrayUnion.40"] }
%"class.llvm::SMDiagnostic" = type { %"class.llvm::SourceMgr"*, %"class.llvm::SMLoc", %"class.std::basic_string", i32, i32, i32, %"class.std::basic_string", %"class.std::basic_string", %"class.std::vector.79", %"class.llvm::SmallVector.84" }
%"class.llvm::SourceMgr" = type { %"class.std::vector", %"class.std::vector.74", i8*, void (%"class.llvm::SMDiagnostic"*, i8*)*, i8* }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<llvm::SourceMgr::SrcBuffer, std::allocator<llvm::SourceMgr::SrcBuffer> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::SourceMgr::SrcBuffer, std::allocator<llvm::SourceMgr::SrcBuffer> >::_Vector_impl" = type { %"struct.llvm::SourceMgr::SrcBuffer"*, %"struct.llvm::SourceMgr::SrcBuffer"*, %"struct.llvm::SourceMgr::SrcBuffer"* }
%"struct.llvm::SourceMgr::SrcBuffer" = type { %"class.llvm::MemoryBuffer"*, %"class.llvm::SMLoc" }
%"class.llvm::MemoryBuffer" = type { i32 (...)**, i8*, i8* }
%"class.std::vector.74" = type { %"struct.std::_Vector_base.75" }
%"struct.std::_Vector_base.75" = type { %"struct.std::_Vector_base<std::basic_string<char>, std::allocator<std::basic_string<char> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::basic_string<char>, std::allocator<std::basic_string<char> > >::_Vector_impl" = type { %"class.std::basic_string"*, %"class.std::basic_string"*, %"class.std::basic_string"* }
%"class.llvm::SMLoc" = type { i8* }
%"class.std::vector.79" = type { %"struct.std::_Vector_base.80" }
%"struct.std::_Vector_base.80" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>, std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>, std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" = type { %"struct.std::pair"*, %"struct.std::pair"*, %"struct.std::pair"* }
%"struct.std::pair" = type { i32, i32 }
%"class.llvm::SmallVector.84" = type { %"class.llvm::SmallVectorImpl.85", %"struct.llvm::SmallVectorStorage.90" }
%"class.llvm::SmallVectorImpl.85" = type { %"class.llvm::SmallVectorTemplateBase.86" }
%"class.llvm::SmallVectorTemplateBase.86" = type { %"class.llvm::SmallVectorTemplateCommon.87" }
%"class.llvm::SmallVectorTemplateCommon.87" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.88" }
%"struct.llvm::AlignedCharArrayUnion.88" = type { %"struct.llvm::AlignedCharArray.89" }
%"struct.llvm::AlignedCharArray.89" = type { [24 x i8] }
%"struct.llvm::SmallVectorStorage.90" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.88"] }
%"class.llvm::LLVMContext" = type { %"class.llvm::LLVMContextImpl"* }
%"class.llvm::LLVMContextImpl" = type opaque
%"class.std::allocator" = type { i8 }
%"class.llvm::ErrorOr.109" = type { %union.anon.110, i8, [7 x i8] }
%union.anon.110 = type { %"struct.llvm::AlignedCharArrayUnion.93" }
%"struct.llvm::AlignedCharArrayUnion.93" = type { %"struct.llvm::AlignedCharArray.94" }
%"struct.llvm::AlignedCharArray.94" = type { [16 x i8] }
%"class.llvm::ErrorOr" = type { %union.anon, i8, [7 x i8] }
%union.anon = type { %"struct.llvm::AlignedCharArrayUnion.93" }
%"class.std::error_category" = type { i32 (...)** }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep_base" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep_base" = type { i64, i64, i32 }
%"class.llvm::SMFixIt" = type { %"class.llvm::SMRange", %"class.std::basic_string" }
%"class.llvm::SMRange" = type { %"class.llvm::SMLoc", %"class.llvm::SMLoc" }
%"struct.llvm::NamedRegionTimer" = type { %"class.llvm::TimeRegion" }
%"class.llvm::TimeRegion" = type { %"class.llvm::Timer"* }
%"class.llvm::Timer" = type { %"class.llvm::TimeRecord", %"class.std::basic_string", i8, %"class.llvm::TimerGroup"*, %"class.llvm::Timer"**, %"class.llvm::Timer"* }
%"class.llvm::TimeRecord" = type { double, double, double, i64 }
%"class.llvm::TimerGroup" = type { %"class.std::basic_string", %"class.llvm::Timer"*, %"class.std::vector.103", %"class.llvm::TimerGroup"**, %"class.llvm::TimerGroup"* }
%"class.std::vector.103" = type { %"struct.std::_Vector_base.104" }
%"struct.std::_Vector_base.104" = type { %"struct.std::_Vector_base<std::pair<llvm::TimeRecord, std::basic_string<char> >, std::allocator<std::pair<llvm::TimeRecord, std::basic_string<char> > > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<llvm::TimeRecord, std::basic_string<char> >, std::allocator<std::pair<llvm::TimeRecord, std::basic_string<char> > > >::_Vector_impl" = type { %"struct.std::pair.108"*, %"struct.std::pair.108"*, %"struct.std::pair.108"* }
%"struct.std::pair.108" = type opaque
%struct.LLVMOpaqueContext = type opaque
%struct.LLVMOpaqueMemoryBuffer = type opaque
%struct.LLVMOpaqueModule = type opaque
%"class.llvm::raw_string_ostream" = type { %"class.llvm::raw_ostream.base", %"class.std::basic_string"* }
%"class.llvm::raw_ostream.base" = type <{ i32 (...)**, i8*, i8*, i8*, i32 }>
%"class.llvm::raw_ostream" = type { i32 (...)**, i8*, i8*, i8*, i32 }

@.str = private unnamed_addr constant [28 x i8] c"Could not open input file: \00", align 1
@.str1 = private unnamed_addr constant [54 x i8] c"!HasError && \22Cannot get value when an error exists!\22\00", align 1
@.str2 = private unnamed_addr constant [61 x i8] c"/home/wschmidt/llvm/llvm-test/include/llvm/Support/ErrorOr.h\00", align 1
@__PRETTY_FUNCTION__._ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE10getStorageEv = private unnamed_addr constant [206 x i8] c"storage_type *llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > >::getStorage() [T = std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >]\00", align 1
@_ZNSs4_Rep20_S_empty_rep_storageE = external global [0 x i64]

declare void @_ZN4llvm12MemoryBuffer14getFileOrSTDINENS_9StringRefEl(%"class.llvm::ErrorOr"* sret, [2 x i64], i64) #1

declare void @_ZN4llvm16NamedRegionTimerC1ENS_9StringRefES1_b(%"struct.llvm::NamedRegionTimer"*, [2 x i64], [2 x i64], i1 zeroext) #1

; Function Attrs: nounwind
define %"class.llvm::Module"* @_ZN4llvm11ParseIRFileERKSsRNS_12SMDiagnosticERNS_11LLVMContextE(%"class.std::basic_string"* nocapture readonly dereferenceable(8) %Filename, %"class.llvm::SMDiagnostic"* dereferenceable(200) %Err, %"class.llvm::LLVMContext"* dereferenceable(8) %Context) #0 {
entry:
; CHECK: .globl	_ZN4llvm11ParseIRFileERKSsRNS_12SMDiagnosticERNS_11LLVMContextE
; CHECK: bctrl
; CHECK: ld 2, 24(1)
; CHECK: addis [[REG:[0-9]+]], 2, .L.str@toc@ha
; CHECK: addi {{[0-9]+}}, [[REG]], .L.str@toc@l
; CHECK: bl _ZNSs6insertEmPKcm
  %.atomicdst.i.i.i.i.i46 = alloca i32, align 4
  %ref.tmp.i.i47 = alloca %"class.std::allocator", align 1
  %.atomicdst.i.i.i.i.i = alloca i32, align 4
  %ref.tmp.i.i = alloca %"class.std::allocator", align 1
  %ref.tmp.i.i2.i = alloca %"class.std::allocator", align 1
  %ref.tmp.i.i.i = alloca %"class.std::allocator", align 1
  %FileOrErr = alloca %"class.llvm::ErrorOr", align 8
  %ref.tmp = alloca %"class.llvm::SMDiagnostic", align 8
  %ref.tmp5 = alloca %"class.std::basic_string", align 8
  %_M_p.i.i.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %Filename, i64 0, i32 0, i32 0
  %0 = load i8*, i8** %_M_p.i.i.i, align 8, !tbaa !1
  %1 = ptrtoint i8* %0 to i64
  %arrayidx.i.i.i = getelementptr inbounds i8, i8* %0, i64 -24
  %_M_length.i.i = bitcast i8* %arrayidx.i.i.i to i64*
  %2 = load i64, i64* %_M_length.i.i, align 8, !tbaa !7
  %.fca.0.insert18 = insertvalue [2 x i64] undef, i64 %1, 0
  %.fca.1.insert21 = insertvalue [2 x i64] %.fca.0.insert18, i64 %2, 1
  call void @_ZN4llvm12MemoryBuffer14getFileOrSTDINENS_9StringRefEl(%"class.llvm::ErrorOr"* sret %FileOrErr, [2 x i64] %.fca.1.insert21, i64 -1) #3
  %HasError.i24 = getelementptr inbounds %"class.llvm::ErrorOr", %"class.llvm::ErrorOr"* %FileOrErr, i64 0, i32 1
  %bf.load.i25 = load i8, i8* %HasError.i24, align 8
  %3 = and i8 %bf.load.i25, 1
  %bf.cast.i26 = icmp eq i8 %3, 0
  br i1 %bf.cast.i26, label %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE3getEv.exit, label %_ZNK4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE8getErrorEv.exit

_ZNK4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE8getErrorEv.exit: ; preds = %entry
  %retval.sroa.0.0..sroa_cast.i = bitcast %"class.llvm::ErrorOr"* %FileOrErr to i64*
  %retval.sroa.0.0.copyload.i = load i64, i64* %retval.sroa.0.0..sroa_cast.i, align 8
  %retval.sroa.3.0..sroa_idx.i = getelementptr inbounds %"class.llvm::ErrorOr", %"class.llvm::ErrorOr"* %FileOrErr, i64 0, i32 0, i32 0, i32 0, i32 0, i64 8
  %retval.sroa.3.0..sroa_cast.i = bitcast i8* %retval.sroa.3.0..sroa_idx.i to i64*
  %retval.sroa.3.0.copyload.i = load i64, i64* %retval.sroa.3.0..sroa_cast.i, align 8
  %phitmp = trunc i64 %retval.sroa.0.0.copyload.i to i32
  %cmp.i = icmp eq i32 %phitmp, 0
  br i1 %cmp.i, label %cond.false.i.i, label %if.then

if.then:                                          ; preds = %_ZNK4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE8getErrorEv.exit
  %.c = inttoptr i64 %retval.sroa.3.0.copyload.i to %"class.std::error_category"*
  %4 = load i8*, i8** %_M_p.i.i.i, align 8, !tbaa !1
  %arrayidx.i.i.i30 = getelementptr inbounds i8, i8* %4, i64 -24
  %_M_length.i.i31 = bitcast i8* %arrayidx.i.i.i30 to i64*
  %5 = load i64, i64* %_M_length.i.i31, align 8, !tbaa !7
  %6 = inttoptr i64 %retval.sroa.3.0.copyload.i to void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)***
  %vtable.i = load void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)**, void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)*** %6, align 8, !tbaa !11
  %vfn.i = getelementptr inbounds void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)*, void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)** %vtable.i, i64 3
  %7 = load void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)*, void (%"class.std::basic_string"*, %"class.std::error_category"*, i32)** %vfn.i, align 8
  call void %7(%"class.std::basic_string"* sret %ref.tmp5, %"class.std::error_category"* %.c, i32 signext %phitmp) #3
  %call2.i.i = call dereferenceable(8) %"class.std::basic_string"* @_ZNSs6insertEmPKcm(%"class.std::basic_string"* %ref.tmp5, i64 0, i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str, i64 0, i64 0), i64 27) #3
  %_M_p2.i.i.i.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %call2.i.i, i64 0, i32 0, i32 0
  %8 = load i8*, i8** %_M_p2.i.i.i.i, align 8, !tbaa !13
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p2.i.i.i.i, align 8, !tbaa !1
  %arrayidx.i.i.i36 = getelementptr inbounds i8, i8* %8, i64 -24
  %_M_length.i.i37 = bitcast i8* %arrayidx.i.i.i36 to i64*
  %9 = load i64, i64* %_M_length.i.i37, align 8, !tbaa !7
  %Filename.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 2
  %10 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i.i2.i, i64 0, i32 0
  %11 = bitcast %"class.llvm::SMDiagnostic"* %ref.tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %11, i8 0, i64 16, i1 false) #3
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %10) #3
  %tobool.i.i4.i = icmp eq i8* %4, null
  br i1 %tobool.i.i4.i, label %if.then.i.i6.i, label %if.end.i.i8.i

if.then.i.i6.i:                                   ; preds = %if.then
  %_M_p.i.i.i.i.i.i5.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %Filename.i, i64 0, i32 0, i32 0
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p.i.i.i.i.i.i5.i, align 8, !tbaa !13
  br label %_ZNK4llvm9StringRefcvSsEv.exit9.i

if.end.i.i8.i:                                    ; preds = %if.then
  call void @_ZNSsC1EPKcmRKSaIcE(%"class.std::basic_string"* %Filename.i, i8* %4, i64 %5, %"class.std::allocator"* dereferenceable(1) %ref.tmp.i.i2.i) #3
  br label %_ZNK4llvm9StringRefcvSsEv.exit9.i

_ZNK4llvm9StringRefcvSsEv.exit9.i:                ; preds = %if.end.i.i8.i, %if.then.i.i6.i
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %10) #3
  %LineNo.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 3
  store i32 -1, i32* %LineNo.i, align 8, !tbaa !14
  %ColumnNo.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 4
  store i32 -1, i32* %ColumnNo.i, align 4, !tbaa !21
  %Kind.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 5
  store i32 0, i32* %Kind.i, align 8, !tbaa !22
  %Message.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 6
  %12 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i.i.i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %12) #3
  %tobool.i.i.i = icmp eq i8* %8, null
  br i1 %tobool.i.i.i, label %if.then.i.i.i, label %if.end.i.i.i

if.then.i.i.i:                                    ; preds = %_ZNK4llvm9StringRefcvSsEv.exit9.i
  %_M_p.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %Message.i, i64 0, i32 0, i32 0
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p.i.i.i.i.i.i.i, align 8, !tbaa !13
  br label %_ZN4llvm12SMDiagnosticC2ENS_9StringRefENS_9SourceMgr8DiagKindES1_.exit

if.end.i.i.i:                                     ; preds = %_ZNK4llvm9StringRefcvSsEv.exit9.i
  call void @_ZNSsC1EPKcmRKSaIcE(%"class.std::basic_string"* %Message.i, i8* %8, i64 %9, %"class.std::allocator"* dereferenceable(1) %ref.tmp.i.i.i) #3
  br label %_ZN4llvm12SMDiagnosticC2ENS_9StringRefENS_9SourceMgr8DiagKindES1_.exit

_ZN4llvm12SMDiagnosticC2ENS_9StringRefENS_9SourceMgr8DiagKindES1_.exit: ; preds = %if.then.i.i.i, %if.end.i.i.i
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %12) #3
  %_M_p.i.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 7, i32 0, i32 0
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p.i.i.i.i.i, align 8, !tbaa !13
  %Ranges.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 8
  %13 = bitcast %"class.std::vector.79"* %Ranges.i to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %13, i8 0, i64 24, i1 false) #3
  %14 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 9, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %BeginX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 9, i32 0, i32 0, i32 0, i32 0, i32 0
  store i8* %14, i8** %BeginX.i.i.i.i.i.i, align 8, !tbaa !23
  %EndX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 9, i32 0, i32 0, i32 0, i32 0, i32 1
  store i8* %14, i8** %EndX.i.i.i.i.i.i, align 8, !tbaa !25
  %CapacityX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 9, i32 0, i32 0, i32 0, i32 0, i32 2
  %add.ptr.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 9, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 96
  store i8* %add.ptr.i.i.i.i.i.i, i8** %CapacityX.i.i.i.i.i.i, align 8, !tbaa !26
  %15 = bitcast %"class.llvm::SMDiagnostic"* %Err to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %15, i8* align 8 %11, i64 16, i1 false) #3
  %Filename.i38 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 2
  call void @_ZNSs4swapERSs(%"class.std::basic_string"* %Filename.i38, %"class.std::basic_string"* dereferenceable(8) %Filename.i) #3
  %LineNo.i39 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 3
  %16 = bitcast i32* %LineNo.i39 to i8*
  %17 = bitcast i32* %LineNo.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %16, i8* align 4 %17, i64 12, i1 false) #3
  %Message.i40 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 6
  call void @_ZNSs4swapERSs(%"class.std::basic_string"* %Message.i40, %"class.std::basic_string"* dereferenceable(8) %Message.i) #3
  %LineContents.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 7
  %LineContents7.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 7
  call void @_ZNSs4swapERSs(%"class.std::basic_string"* %LineContents.i, %"class.std::basic_string"* dereferenceable(8) %LineContents7.i) #3
  %Ranges.i41 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 8
  %_M_start.i7.i.i.i = getelementptr inbounds %"class.std::vector.79", %"class.std::vector.79"* %Ranges.i41, i64 0, i32 0, i32 0, i32 0
  %18 = load %"struct.std::pair"*, %"struct.std::pair"** %_M_start.i7.i.i.i, align 8, !tbaa !27
  %_M_finish.i9.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 8, i32 0, i32 0, i32 1
  %_M_end_of_storage.i11.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 8, i32 0, i32 0, i32 2
  %_M_start2.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 8, i32 0, i32 0, i32 0
  %19 = bitcast %"class.std::vector.79"* %Ranges.i41 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %19, i8 0, i64 16, i1 false) #3
  %20 = load %"struct.std::pair"*, %"struct.std::pair"** %_M_start2.i.i.i.i, align 8, !tbaa !27
  store %"struct.std::pair"* %20, %"struct.std::pair"** %_M_start.i7.i.i.i, align 8, !tbaa !27
  store %"struct.std::pair"* null, %"struct.std::pair"** %_M_start2.i.i.i.i, align 8, !tbaa !27
  %_M_finish3.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 8, i32 0, i32 0, i32 1
  %21 = load %"struct.std::pair"*, %"struct.std::pair"** %_M_finish3.i.i.i.i, align 8, !tbaa !27
  store %"struct.std::pair"* %21, %"struct.std::pair"** %_M_finish.i9.i.i.i, align 8, !tbaa !27
  store %"struct.std::pair"* null, %"struct.std::pair"** %_M_finish3.i.i.i.i, align 8, !tbaa !27
  %_M_end_of_storage4.i.i.i.i = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 8, i32 0, i32 0, i32 2
  %22 = load %"struct.std::pair"*, %"struct.std::pair"** %_M_end_of_storage4.i.i.i.i, align 8, !tbaa !27
  store %"struct.std::pair"* %22, %"struct.std::pair"** %_M_end_of_storage.i11.i.i.i, align 8, !tbaa !27
  store %"struct.std::pair"* null, %"struct.std::pair"** %_M_end_of_storage4.i.i.i.i, align 8, !tbaa !27
  %tobool.i.i.i.i.i.i = icmp eq %"struct.std::pair"* %18, null
  br i1 %tobool.i.i.i.i.i.i, label %_ZN4llvm12SMDiagnosticaSEOS0_.exit, label %if.then.i.i.i.i.i.i

if.then.i.i.i.i.i.i:                              ; preds = %_ZN4llvm12SMDiagnosticC2ENS_9StringRefENS_9SourceMgr8DiagKindES1_.exit
  %23 = bitcast %"struct.std::pair"* %18 to i8*
  call void @_ZdlPv(i8* %23) #3
  br label %_ZN4llvm12SMDiagnosticaSEOS0_.exit

_ZN4llvm12SMDiagnosticaSEOS0_.exit:               ; preds = %_ZN4llvm12SMDiagnosticC2ENS_9StringRefENS_9SourceMgr8DiagKindES1_.exit, %if.then.i.i.i.i.i.i
  %24 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %Err, i64 0, i32 9, i32 0
  %25 = getelementptr inbounds %"class.llvm::SMDiagnostic", %"class.llvm::SMDiagnostic"* %ref.tmp, i64 0, i32 9, i32 0
  %call2.i.i42 = call dereferenceable(48) %"class.llvm::SmallVectorImpl.85"* @_ZN4llvm15SmallVectorImplINS_7SMFixItEEaSEOS2_(%"class.llvm::SmallVectorImpl.85"* %24, %"class.llvm::SmallVectorImpl.85"* dereferenceable(48) %25) #3
  call void @_ZN4llvm12SMDiagnosticD2Ev(%"class.llvm::SMDiagnostic"* %ref.tmp) #3
  %26 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i.i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %26) #3
  %27 = bitcast i8* %arrayidx.i.i.i36 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %cmp.i.i.i = icmp eq i8* %arrayidx.i.i.i36, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i.i, label %_ZNSsD1Ev.exit, label %if.then.i.i.i45, !prof !28

if.then.i.i.i45:                                  ; preds = %_ZN4llvm12SMDiagnosticaSEOS0_.exit
  %_M_refcount.i.i.i = getelementptr inbounds i8, i8* %8, i64 -8
  %28 = bitcast i8* %_M_refcount.i.i.i to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i.i, label %if.else.i.i.i.i

if.then.i.i.i.i:                                  ; preds = %if.then.i.i.i45
  %.atomicdst.i.i.i.i.i.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..sroa_cast = bitcast i32* %.atomicdst.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %.atomicdst.i.i.i.i.i.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..sroa_cast)
  %29 = atomicrmw volatile add i32* %28, i32 -1 acq_rel
  store i32 %29, i32* %.atomicdst.i.i.i.i.i, align 4
  %.atomicdst.i.i.i.i.i.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..atomicdst.0..atomicdst.0..i.i.i.i.i = load volatile i32, i32* %.atomicdst.i.i.i.i.i, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %.atomicdst.i.i.i.i.i.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..sroa_cast)
  br label %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i

if.else.i.i.i.i:                                  ; preds = %if.then.i.i.i45
  %30 = load i32, i32* %28, align 4, !tbaa !29
  %add.i.i.i.i.i = add nsw i32 %30, -1
  store i32 %add.i.i.i.i.i, i32* %28, align 4, !tbaa !29
  br label %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i

_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i: ; preds = %if.else.i.i.i.i, %if.then.i.i.i.i
  %retval.0.i.i.i.i = phi i32 [ %.atomicdst.i.i.i.i.i.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..atomicdst.0..atomicdst.0..i.i.i.i.i, %if.then.i.i.i.i ], [ %30, %if.else.i.i.i.i ]
  %cmp3.i.i.i = icmp slt i32 %retval.0.i.i.i.i, 1
  br i1 %cmp3.i.i.i, label %if.then4.i.i.i, label %_ZNSsD1Ev.exit

if.then4.i.i.i:                                   ; preds = %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* %27, %"class.std::allocator"* dereferenceable(1) %ref.tmp.i.i) #3
  br label %_ZNSsD1Ev.exit

_ZNSsD1Ev.exit:                                   ; preds = %_ZN4llvm12SMDiagnosticaSEOS0_.exit, %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i, %if.then4.i.i.i
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %26) #3
  %31 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i.i47, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %31) #3
  %_M_p.i.i.i.i48 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp5, i64 0, i32 0, i32 0
  %32 = load i8*, i8** %_M_p.i.i.i.i48, align 8, !tbaa !1
  %arrayidx.i.i.i49 = getelementptr inbounds i8, i8* %32, i64 -24
  %33 = bitcast i8* %arrayidx.i.i.i49 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %cmp.i.i.i50 = icmp eq i8* %arrayidx.i.i.i49, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i.i50, label %_ZNSsD1Ev.exit62, label %if.then.i.i.i52, !prof !28

if.then.i.i.i52:                                  ; preds = %_ZNSsD1Ev.exit
  %_M_refcount.i.i.i51 = getelementptr inbounds i8, i8* %32, i64 -8
  %34 = bitcast i8* %_M_refcount.i.i.i51 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i.i55, label %if.else.i.i.i.i57

if.then.i.i.i.i55:                                ; preds = %if.then.i.i.i52
  %.atomicdst.i.i.i.i.i46.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..sroa_cast = bitcast i32* %.atomicdst.i.i.i.i.i46 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %.atomicdst.i.i.i.i.i46.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..sroa_cast)
  %35 = atomicrmw volatile add i32* %34, i32 -1 acq_rel
  store i32 %35, i32* %.atomicdst.i.i.i.i.i46, align 4
  %.atomicdst.i.i.i.i.i46.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..atomicdst.0..atomicdst.0..i.i.i.i.i54 = load volatile i32, i32* %.atomicdst.i.i.i.i.i46, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %.atomicdst.i.i.i.i.i46.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..sroa_cast)
  br label %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i60

if.else.i.i.i.i57:                                ; preds = %if.then.i.i.i52
  %36 = load i32, i32* %34, align 4, !tbaa !29
  %add.i.i.i.i.i56 = add nsw i32 %36, -1
  store i32 %add.i.i.i.i.i56, i32* %34, align 4, !tbaa !29
  br label %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i60

_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i60: ; preds = %if.else.i.i.i.i57, %if.then.i.i.i.i55
  %retval.0.i.i.i.i58 = phi i32 [ %.atomicdst.i.i.i.i.i46.0..atomicdst.i.i.i.i.0..atomicdst.i.i.i.0..atomicdst.i.i.0..atomicdst.i.0..atomicdst.0..atomicdst.0..i.i.i.i.i54, %if.then.i.i.i.i55 ], [ %36, %if.else.i.i.i.i57 ]
  %cmp3.i.i.i59 = icmp slt i32 %retval.0.i.i.i.i58, 1
  br i1 %cmp3.i.i.i59, label %if.then4.i.i.i61, label %_ZNSsD1Ev.exit62

if.then4.i.i.i61:                                 ; preds = %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i60
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* %33, %"class.std::allocator"* dereferenceable(1) %ref.tmp.i.i47) #3
  br label %_ZNSsD1Ev.exit62

_ZNSsD1Ev.exit62:                                 ; preds = %_ZNSsD1Ev.exit, %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i.i60, %if.then4.i.i.i61
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %31) #3
  br label %cleanup

cond.false.i.i:                                   ; preds = %_ZNK4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE8getErrorEv.exit
  call void @__assert_fail(i8* getelementptr inbounds ([54 x i8], [54 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([61 x i8], [61 x i8]* @.str2, i64 0, i64 0), i32 zeroext 242, i8* getelementptr inbounds ([206 x i8], [206 x i8]* @__PRETTY_FUNCTION__._ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE10getStorageEv, i64 0, i64 0)) #7
  unreachable

_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE3getEv.exit: ; preds = %entry
  %_M_head_impl.i.i.i.i.i = bitcast %"class.llvm::ErrorOr"* %FileOrErr to %"class.llvm::MemoryBuffer"**
  %37 = load %"class.llvm::MemoryBuffer"*, %"class.llvm::MemoryBuffer"** %_M_head_impl.i.i.i.i.i, align 8, !tbaa !27
  %call9 = call %"class.llvm::Module"* @_ZN4llvm7ParseIREPNS_12MemoryBufferERNS_12SMDiagnosticERNS_11LLVMContextE(%"class.llvm::MemoryBuffer"* %37, %"class.llvm::SMDiagnostic"* dereferenceable(200) %Err, %"class.llvm::LLVMContext"* dereferenceable(8) %Context)
  br label %cleanup

cleanup:                                          ; preds = %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE3getEv.exit, %_ZNSsD1Ev.exit62
  %retval.0 = phi %"class.llvm::Module"* [ null, %_ZNSsD1Ev.exit62 ], [ %call9, %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE3getEv.exit ]
  %bf.load.i = load i8, i8* %HasError.i24, align 8
  %38 = and i8 %bf.load.i, 1
  %bf.cast.i = icmp eq i8 %38, 0
  br i1 %bf.cast.i, label %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE10getStorageEv.exit.i, label %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEED2Ev.exit

_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE10getStorageEv.exit.i: ; preds = %cleanup
  %_M_head_impl.i.i.i.i.i.i = bitcast %"class.llvm::ErrorOr"* %FileOrErr to %"class.llvm::MemoryBuffer"**
  %39 = load %"class.llvm::MemoryBuffer"*, %"class.llvm::MemoryBuffer"** %_M_head_impl.i.i.i.i.i.i, align 8, !tbaa !27
  %cmp.i.i = icmp eq %"class.llvm::MemoryBuffer"* %39, null
  br i1 %cmp.i.i, label %_ZNSt10unique_ptrIN4llvm12MemoryBufferESt14default_deleteIS1_EED2Ev.exit.i, label %_ZNKSt14default_deleteIN4llvm12MemoryBufferEEclEPS1_.exit.i.i

_ZNKSt14default_deleteIN4llvm12MemoryBufferEEclEPS1_.exit.i.i: ; preds = %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE10getStorageEv.exit.i
  %40 = bitcast %"class.llvm::MemoryBuffer"* %39 to void (%"class.llvm::MemoryBuffer"*)***
  %vtable.i.i.i = load void (%"class.llvm::MemoryBuffer"*)**, void (%"class.llvm::MemoryBuffer"*)*** %40, align 8, !tbaa !11
  %vfn.i.i.i = getelementptr inbounds void (%"class.llvm::MemoryBuffer"*)*, void (%"class.llvm::MemoryBuffer"*)** %vtable.i.i.i, i64 1
  %41 = load void (%"class.llvm::MemoryBuffer"*)*, void (%"class.llvm::MemoryBuffer"*)** %vfn.i.i.i, align 8
  call void %41(%"class.llvm::MemoryBuffer"* %39) #3
  br label %_ZNSt10unique_ptrIN4llvm12MemoryBufferESt14default_deleteIS1_EED2Ev.exit.i

_ZNSt10unique_ptrIN4llvm12MemoryBufferESt14default_deleteIS1_EED2Ev.exit.i: ; preds = %_ZNKSt14default_deleteIN4llvm12MemoryBufferEEclEPS1_.exit.i.i, %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEE10getStorageEv.exit.i
  store %"class.llvm::MemoryBuffer"* null, %"class.llvm::MemoryBuffer"** %_M_head_impl.i.i.i.i.i.i, align 8, !tbaa !27
  br label %_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEED2Ev.exit

_ZN4llvm7ErrorOrISt10unique_ptrINS_12MemoryBufferESt14default_deleteIS2_EEED2Ev.exit: ; preds = %cleanup, %_ZNSt10unique_ptrIN4llvm12MemoryBufferESt14default_deleteIS1_EED2Ev.exit.i
  ret %"class.llvm::Module"* %retval.0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #3

; Function Attrs: noreturn nounwind
declare void @__assert_fail(i8*, i8*, i32 zeroext, i8*) #4

declare dereferenceable(8) %"class.std::basic_string"* @_ZNSs6insertEmPKcm(%"class.std::basic_string"*, i64, i8*, i64) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #3

; Function Attrs: nounwind
declare void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*, %"class.std::allocator"* dereferenceable(1)) #0

; Function Attrs: nounwind
declare extern_weak signext i32 @__pthread_key_create(i32*, void (i8*)*) #0

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #6

declare void @_ZNSsC1EPKcmRKSaIcE(%"class.std::basic_string"*, i8*, i64, %"class.std::allocator"* dereferenceable(1)) #1

declare hidden void @_ZN4llvm12SMDiagnosticD2Ev(%"class.llvm::SMDiagnostic"* readonly %this) unnamed_addr #2 align 2

declare dereferenceable(48) %"class.llvm::SmallVectorImpl.85"* @_ZN4llvm15SmallVectorImplINS_7SMFixItEEaSEOS2_(%"class.llvm::SmallVectorImpl.85"* %this, %"class.llvm::SmallVectorImpl.85"* dereferenceable(48) %RHS) #0 align 2

declare %"class.llvm::Module"* @_ZN4llvm7ParseIREPNS_12MemoryBufferERNS_12SMDiagnosticERNS_11LLVMContextE(%"class.llvm::MemoryBuffer"* %Buffer, %"class.llvm::SMDiagnostic"* dereferenceable(200) %Err, %"class.llvm::LLVMContext"* dereferenceable(8) %Context) #0

declare void @_ZNSs4swapERSs(%"class.std::basic_string"*, %"class.std::basic_string"* dereferenceable(8)) #1

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readonly "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nobuiltin nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.6.0 (trunk 215115) (llvm/trunk 215117)"}
!1 = !{!2, !4, i64 0}
!2 = !{!"_ZTSSs", !3, i64 0}
!3 = !{!"_ZTSNSs12_Alloc_hiderE", !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !9, i64 0}
!8 = !{!"_ZTSNSs9_Rep_baseE", !9, i64 0, !9, i64 8, !10, i64 16}
!9 = !{!"long", !5, i64 0}
!10 = !{!"int", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"vtable pointer", !6, i64 0}
!13 = !{!3, !4, i64 0}
!14 = !{!15, !10, i64 24}
!15 = !{!"_ZTSN4llvm12SMDiagnosticE", !4, i64 0, !16, i64 8, !2, i64 16, !10, i64 24, !10, i64 28, !17, i64 32, !2, i64 40, !2, i64 48, !18, i64 56, !19, i64 80}
!16 = !{!"_ZTSN4llvm5SMLocE", !4, i64 0}
!17 = !{!"_ZTSN4llvm9SourceMgr8DiagKindE", !5, i64 0}
!18 = !{!"_ZTSSt6vectorISt4pairIjjESaIS1_EE"}
!19 = !{!"_ZTSN4llvm11SmallVectorINS_7SMFixItELj4EEE", !20, i64 48}
!20 = !{!"_ZTSN4llvm18SmallVectorStorageINS_7SMFixItELj4EEE", !5, i64 0}
!21 = !{!15, !10, i64 28}
!22 = !{!15, !17, i64 32}
!23 = !{!24, !4, i64 0}
!24 = !{!"_ZTSN4llvm15SmallVectorBaseE", !4, i64 0, !4, i64 8, !4, i64 16}
!25 = !{!24, !4, i64 8}
!26 = !{!24, !4, i64 16}
!27 = !{!4, !4, i64 0}
!28 = !{!"branch_weights", i32 64, i32 4}
!29 = !{!10, !10, i64 0}
!30 = !{!31, !4, i64 8}
!31 = !{!"_ZTSN4llvm12MemoryBufferE", !4, i64 8, !4, i64 16}
!32 = !{!31, !4, i64 16}
!33 = !{!5, !5, i64 0}
!34 = !{!35, !4, i64 0}
!35 = !{!"_ZTSSt12_Vector_baseISt4pairIjjESaIS1_EE", !36, i64 0}
!36 = !{!"_ZTSNSt12_Vector_baseISt4pairIjjESaIS1_EE12_Vector_implE", !4, i64 0, !4, i64 8, !4, i64 16}
!37 = !{!38, !38, i64 0}
!38 = !{!"bool", !5, i64 0}
!39 = !{i8 0, i8 2}
!40 = !{!41, !4, i64 0}
!41 = !{!"_ZTSN4llvm10TimeRegionE", !4, i64 0}
!42 = !{!43, !44, i64 32}
!43 = !{!"_ZTSN4llvm11raw_ostreamE", !4, i64 8, !4, i64 16, !4, i64 24, !44, i64 32}
!44 = !{!"_ZTSN4llvm11raw_ostream10BufferKindE", !5, i64 0}
!45 = !{!43, !4, i64 24}
!46 = !{!43, !4, i64 8}
!47 = !{i64 0, i64 8, !27, i64 8, i64 8, !27}
