; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O3 < %s | FileCheck %s

; Test case derived from bug report 15031.  The code in the post-RA
; scheduler to break critical anti-dependencies was failing to check
; whether an instruction had more than one definition, and ensuring
; that any additional definitions interfered with the choice of a new
; register.  As a result, this test originally caused this to be
; generated:
;
;   lbzu 3, 1(3)
;
; which is illegal, since it requires register 3 to both receive the
; loaded value and receive the updated address.  With the fix to bug
; 15031, a different register is chosen to receive the loaded value.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%"class.llvm::MachineMemOperand" = type { %"struct.llvm::MachinePointerInfo", i64, i32, ptr, ptr }
%"struct.llvm::MachinePointerInfo" = type { ptr, i64 }
%"class.llvm::Value" = type { ptr, i8, i8, i16, ptr, ptr, ptr }
%"class.llvm::Type" = type { ptr, i32, i32, ptr }
%"class.llvm::LLVMContext" = type { ptr }
%"class.llvm::LLVMContextImpl" = type opaque
%"class.llvm::Use" = type { ptr, ptr, %"class.llvm::PointerIntPair" }
%"class.llvm::PointerIntPair" = type { i64 }
%"class.llvm::StringMapEntry" = type opaque
%"class.llvm::MDNode" = type { %"class.llvm::Value", %"class.llvm::FoldingSetImpl::Node", i32, i32 }
%"class.llvm::FoldingSetImpl::Node" = type { ptr }
%"class.llvm::MachineInstr" = type { %"class.llvm::ilist_node", ptr, ptr, ptr, i32, %"class.llvm::ArrayRecycler<llvm::MachineOperand, 8>::Capacity", i8, i8, i8, ptr, %"class.llvm::DebugLoc" }
%"class.llvm::ilist_node" = type { %"class.llvm::ilist_half_node", ptr }
%"class.llvm::ilist_half_node" = type { ptr }
%"class.llvm::MCInstrDesc" = type { i16, i16, i16, i16, i16, i32, i64, ptr, ptr, ptr }
%"class.llvm::MCOperandInfo" = type { i16, i8, i8, i32 }
%"class.llvm::MachineBasicBlock" = type { %"class.llvm::ilist_node.0", %"struct.llvm::ilist", ptr, i32, ptr, %"class.std::vector.163", %"class.std::vector.163", %"class.std::vector.123", %"class.std::vector.123", i32, i8, i8 }
%"class.llvm::ilist_node.0" = type { %"class.llvm::ilist_half_node.1", ptr }
%"class.llvm::ilist_half_node.1" = type { ptr }
%"struct.llvm::ilist" = type { %"class.llvm::iplist" }
%"class.llvm::iplist" = type { %"struct.llvm::ilist_traits", ptr }
%"struct.llvm::ilist_traits" = type { %"class.llvm::ilist_half_node", ptr }
%"class.llvm::BasicBlock" = type { %"class.llvm::Value", %"class.llvm::ilist_node.2", %"class.llvm::iplist.4", ptr }
%"class.llvm::ilist_node.2" = type { %"class.llvm::ilist_half_node.3", ptr }
%"class.llvm::ilist_half_node.3" = type { ptr }
%"class.llvm::iplist.4" = type { %"struct.llvm::ilist_traits.5", ptr }
%"struct.llvm::ilist_traits.5" = type { %"class.llvm::ilist_half_node.10" }
%"class.llvm::ilist_half_node.10" = type { ptr }
%"class.llvm::Instruction" = type { %"class.llvm::User", %"class.llvm::ilist_node.193", ptr, %"class.llvm::DebugLoc" }
%"class.llvm::User" = type { %"class.llvm::Value", ptr, i32 }
%"class.llvm::ilist_node.193" = type { %"class.llvm::ilist_half_node.10", ptr }
%"class.llvm::DebugLoc" = type { i32, i32 }
%"class.llvm::Function" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.27", %"class.llvm::iplist.47", %"class.llvm::iplist.54", ptr, %"class.llvm::AttributeSet" }
%"class.llvm::GlobalValue" = type { [52 x i8], [4 x i8], ptr, %"class.std::basic_string" }
%"class.llvm::Module" = type { ptr, %"class.llvm::iplist.11", %"class.llvm::iplist.20", %"class.llvm::iplist.29", %"struct.llvm::ilist.38", %"class.std::basic_string", ptr, %"class.llvm::OwningPtr", %"class.std::basic_string", %"class.std::basic_string", %"class.std::basic_string", ptr }
%"class.llvm::iplist.11" = type { %"struct.llvm::ilist_traits.12", ptr }
%"struct.llvm::ilist_traits.12" = type { %"class.llvm::ilist_node.18" }
%"class.llvm::ilist_node.18" = type { %"class.llvm::ilist_half_node.19", ptr }
%"class.llvm::ilist_half_node.19" = type { ptr }
%"class.llvm::GlobalVariable" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.18", i8 }
%"class.llvm::iplist.20" = type { %"struct.llvm::ilist_traits.21", ptr }
%"struct.llvm::ilist_traits.21" = type { %"class.llvm::ilist_node.27" }
%"class.llvm::ilist_node.27" = type { %"class.llvm::ilist_half_node.28", ptr }
%"class.llvm::ilist_half_node.28" = type { ptr }
%"class.llvm::iplist.29" = type { %"struct.llvm::ilist_traits.30", ptr }
%"struct.llvm::ilist_traits.30" = type { %"class.llvm::ilist_node.36" }
%"class.llvm::ilist_node.36" = type { %"class.llvm::ilist_half_node.37", ptr }
%"class.llvm::ilist_half_node.37" = type { ptr }
%"class.llvm::GlobalAlias" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.36" }
%"struct.llvm::ilist.38" = type { %"class.llvm::iplist.39" }
%"class.llvm::iplist.39" = type { %"struct.llvm::ilist_traits.40", ptr }
%"struct.llvm::ilist_traits.40" = type { %"class.llvm::ilist_node.45" }
%"class.llvm::ilist_node.45" = type { %"class.llvm::ilist_half_node.46", ptr }
%"class.llvm::ilist_half_node.46" = type { ptr }
%"class.llvm::NamedMDNode" = type { %"class.llvm::ilist_node.45", %"class.std::basic_string", ptr, ptr }
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { ptr }
%"class.llvm::ValueSymbolTable" = type opaque
%"class.llvm::OwningPtr" = type { ptr }
%"class.llvm::GVMaterializer" = type opaque
%"class.llvm::iplist.47" = type { %"struct.llvm::ilist_traits.48", ptr }
%"struct.llvm::ilist_traits.48" = type { %"class.llvm::ilist_half_node.3" }
%"class.llvm::iplist.54" = type { %"struct.llvm::ilist_traits.55", ptr }
%"struct.llvm::ilist_traits.55" = type { %"class.llvm::ilist_half_node.61" }
%"class.llvm::ilist_half_node.61" = type { ptr }
%"class.llvm::Argument" = type { %"class.llvm::Value", %"class.llvm::ilist_node.192", ptr }
%"class.llvm::ilist_node.192" = type { %"class.llvm::ilist_half_node.61", ptr }
%"class.llvm::AttributeSet" = type { ptr }
%"class.llvm::AttributeSetImpl" = type opaque
%"class.llvm::MachineFunction" = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %"class.std::vector.163", %"class.llvm::BumpPtrAllocator", %"class.llvm::Recycler", %"class.llvm::ArrayRecycler", %"class.llvm::Recycler.180", %"struct.llvm::ilist.181", i32, i32, i8 }
%"class.llvm::TargetMachine" = type { ptr, ptr, %"class.std::basic_string", %"class.std::basic_string", %"class.std::basic_string", ptr, ptr, i8, %"class.llvm::TargetOptions" }
%"class.llvm::Target" = type opaque
%"class.llvm::MCCodeGenInfo" = type opaque
%"class.llvm::MCAsmInfo" = type opaque
%"class.llvm::TargetOptions" = type { [2 x i8], i32, i8, i32, i8, %"class.std::basic_string", i32, i32 }
%"class.llvm::MCContext" = type { ptr, ptr, ptr, ptr, %"class.llvm::BumpPtrAllocator", %"class.llvm::StringMap", %"class.llvm::StringMap.62", i32, %"class.llvm::DenseMap.63", ptr, ptr, i8, %"class.std::basic_string", %"class.std::basic_string", %"class.std::vector", %"class.std::vector.70", %"class.llvm::MCDwarfLoc", i8, i8, i32, ptr, ptr, ptr, %"class.std::vector.75", %"class.llvm::StringRef", %"class.llvm::StringRef", i8, %"class.llvm::DenseMap.80", %"class.std::vector.84", ptr, ptr, ptr, i8 }
%"class.llvm::SourceMgr" = type opaque
%"class.llvm::MCRegisterInfo" = type { ptr, i32, i32, i32, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, i32, i32, ptr, ptr, ptr, ptr, %"class.llvm::DenseMap" }
%"struct.llvm::MCRegisterDesc" = type { i32, i32, i32, i32, i32, i32 }
%"class.llvm::MCRegisterClass" = type { ptr, ptr, ptr, i16, i16, i16, i16, i16, i8, i8 }
%"struct.llvm::MCRegisterInfo::DwarfLLVMRegPair" = type { i32, i32 }
%"class.llvm::DenseMap" = type { ptr, i32, i32, i32 }
%"struct.std::pair" = type { i32, i32 }
%"class.llvm::MCObjectFileInfo" = type opaque
%"class.llvm::BumpPtrAllocator" = type { i64, i64, ptr, ptr, ptr, ptr, i64 }
%"class.llvm::SlabAllocator" = type { ptr }
%"class.llvm::MemSlab" = type { i64, ptr }
%"class.llvm::StringMap" = type { %"class.llvm::StringMapImpl", ptr }
%"class.llvm::StringMapImpl" = type { ptr, i32, i32, i32, i32 }
%"class.llvm::StringMapEntryBase" = type { i32 }
%"class.llvm::StringMap.62" = type { %"class.llvm::StringMapImpl", ptr }
%"class.llvm::DenseMap.63" = type { ptr, i32, i32, i32 }
%"struct.std::pair.66" = type opaque
%"class.llvm::raw_ostream" = type { ptr, ptr, ptr, ptr, i32 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<llvm::MCDwarfFile *, std::allocator<llvm::MCDwarfFile *> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MCDwarfFile *, std::allocator<llvm::MCDwarfFile *> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MCDwarfFile" = type { %"class.llvm::StringRef", i32 }
%"class.llvm::StringRef" = type { ptr, i64 }
%"class.std::vector.70" = type { %"struct.std::_Vector_base.71" }
%"struct.std::_Vector_base.71" = type { %"struct.std::_Vector_base<llvm::StringRef, std::allocator<llvm::StringRef> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::StringRef, std::allocator<llvm::StringRef> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MCDwarfLoc" = type { i32, i32, i32, i32, i32, i32 }
%"class.llvm::MCSection" = type opaque
%"class.llvm::MCSymbol" = type { %"class.llvm::StringRef", ptr, ptr, i8 }
%"class.llvm::MCExpr" = type opaque
%"class.std::vector.75" = type { %"struct.std::_Vector_base.76" }
%"struct.std::_Vector_base.76" = type { %"struct.std::_Vector_base<const llvm::MCGenDwarfLabelEntry *, std::allocator<const llvm::MCGenDwarfLabelEntry *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::MCGenDwarfLabelEntry *, std::allocator<const llvm::MCGenDwarfLabelEntry *> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MCGenDwarfLabelEntry" = type { %"class.llvm::StringRef", i32, i32, ptr }
%"class.llvm::DenseMap.80" = type { ptr, i32, i32, i32 }
%"struct.std::pair.83" = type { ptr, ptr }
%"class.llvm::MCLineSection" = type { %"class.std::vector.215" }
%"class.std::vector.215" = type { %"struct.std::_Vector_base.216" }
%"struct.std::_Vector_base.216" = type { %"struct.std::_Vector_base<llvm::MCLineEntry, std::allocator<llvm::MCLineEntry> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MCLineEntry, std::allocator<llvm::MCLineEntry> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MCLineEntry" = type { %"class.llvm::MCDwarfLoc", ptr }
%"class.std::vector.84" = type { %"struct.std::_Vector_base.85" }
%"struct.std::_Vector_base.85" = type { %"struct.std::_Vector_base<const llvm::MCSection *, std::allocator<const llvm::MCSection *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::MCSection *, std::allocator<const llvm::MCSection *> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MachineModuleInfo" = type { %"class.llvm::ImmutablePass", %"class.llvm::MCContext", ptr, ptr, %"class.std::vector.95", i32, %"class.std::vector.100", %"class.llvm::DenseMap.110", %"class.llvm::DenseMap.114", i32, %"class.std::vector.118", %"class.std::vector.123", %"class.std::vector.123", %"class.std::vector.128", %"class.llvm::SmallPtrSet", ptr, i8, i8, i8, i8, %"class.llvm::SmallVector.133" }
%"class.llvm::ImmutablePass" = type { %"class.llvm::ModulePass" }
%"class.llvm::ModulePass" = type { %"class.llvm::Pass" }
%"class.llvm::Pass" = type { ptr, ptr, ptr, i32 }
%"class.llvm::AnalysisResolver" = type { %"class.std::vector.89", ptr }
%"class.std::vector.89" = type { %"struct.std::_Vector_base.90" }
%"struct.std::_Vector_base.90" = type { %"struct.std::_Vector_base<std::pair<const ptr, llvm::Pass *>, std::allocator<std::pair<const ptr, llvm::Pass *> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<const ptr, llvm::Pass *>, std::allocator<std::pair<const ptr, llvm::Pass *> > >::_Vector_impl" = type { ptr, ptr, ptr }
%"struct.std::pair.94" = type { ptr, ptr }
%"class.llvm::PMDataManager" = type opaque
%"class.llvm::MachineModuleInfoImpl" = type { ptr }
%"class.std::vector.95" = type { %"struct.std::_Vector_base.96" }
%"struct.std::_Vector_base.96" = type { %"struct.std::_Vector_base<llvm::MachineMove, std::allocator<llvm::MachineMove> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MachineMove, std::allocator<llvm::MachineMove> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MachineMove" = type { ptr, %"class.llvm::MachineLocation", %"class.llvm::MachineLocation" }
%"class.llvm::MachineLocation" = type { i8, i32, i32 }
%"class.std::vector.100" = type { %"struct.std::_Vector_base.101" }
%"struct.std::_Vector_base.101" = type { %"struct.std::_Vector_base<llvm::LandingPadInfo, std::allocator<llvm::LandingPadInfo> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::LandingPadInfo, std::allocator<llvm::LandingPadInfo> >::_Vector_impl" = type { ptr, ptr, ptr }
%"struct.llvm::LandingPadInfo" = type { ptr, %"class.llvm::SmallVector", %"class.llvm::SmallVector", ptr, ptr, %"class.std::vector.105" }
%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl", %"struct.llvm::SmallVectorStorage" }
%"class.llvm::SmallVectorImpl" = type { %"class.llvm::SmallVectorTemplateBase" }
%"class.llvm::SmallVectorTemplateBase" = type { %"class.llvm::SmallVectorTemplateCommon" }
%"class.llvm::SmallVectorTemplateCommon" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion" }
%"class.llvm::SmallVectorBase" = type { ptr, ptr, ptr }
%"struct.llvm::AlignedCharArrayUnion" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::AlignedCharArray" = type { [8 x i8] }
%"struct.llvm::SmallVectorStorage" = type { i8 }
%"class.std::vector.105" = type { %"struct.std::_Vector_base.106" }
%"struct.std::_Vector_base.106" = type { %"struct.std::_Vector_base<int, std::allocator<int> >::_Vector_impl" }
%"struct.std::_Vector_base<int, std::allocator<int> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::DenseMap.110" = type { ptr, i32, i32, i32 }
%"struct.std::pair.113" = type { ptr, %"class.llvm::SmallVector.206" }
%"class.llvm::SmallVector.206" = type { [28 x i8], %"struct.llvm::SmallVectorStorage.207" }
%"struct.llvm::SmallVectorStorage.207" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.198"] }
%"struct.llvm::AlignedCharArrayUnion.198" = type { %"struct.llvm::AlignedCharArray.199" }
%"struct.llvm::AlignedCharArray.199" = type { [4 x i8] }
%"class.llvm::DenseMap.114" = type { ptr, i32, i32, i32 }
%"struct.std::pair.117" = type { ptr, i32 }
%"class.std::vector.118" = type { %"struct.std::_Vector_base.119" }
%"struct.std::_Vector_base.119" = type { %"struct.std::_Vector_base<const llvm::GlobalVariable *, std::allocator<const llvm::GlobalVariable *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::GlobalVariable *, std::allocator<const llvm::GlobalVariable *> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.std::vector.123" = type { %"struct.std::_Vector_base.124" }
%"struct.std::_Vector_base.124" = type { %"struct.std::_Vector_base<unsigned int, std::allocator<unsigned int> >::_Vector_impl" }
%"struct.std::_Vector_base<unsigned int, std::allocator<unsigned int> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.std::vector.128" = type { %"struct.std::_Vector_base.129" }
%"struct.std::_Vector_base.129" = type { %"struct.std::_Vector_base<const llvm::Function *, std::allocator<const llvm::Function *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::Function *, std::allocator<const llvm::Function *> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::SmallPtrSet" = type { %"class.llvm::SmallPtrSetImpl", [33 x ptr] }
%"class.llvm::SmallPtrSetImpl" = type { ptr, ptr, i32, i32, i32 }
%"class.llvm::MMIAddrLabelMap" = type opaque
%"class.llvm::SmallVector.133" = type { %"class.llvm::SmallVectorImpl.134", %"struct.llvm::SmallVectorStorage.139" }
%"class.llvm::SmallVectorImpl.134" = type { %"class.llvm::SmallVectorTemplateBase.135" }
%"class.llvm::SmallVectorTemplateBase.135" = type { %"class.llvm::SmallVectorTemplateCommon.136" }
%"class.llvm::SmallVectorTemplateCommon.136" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.137" }
%"struct.llvm::AlignedCharArrayUnion.137" = type { %"struct.llvm::AlignedCharArray.138" }
%"struct.llvm::AlignedCharArray.138" = type { [40 x i8] }
%"struct.llvm::SmallVectorStorage.139" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.137"] }
%"class.llvm::GCModuleInfo" = type opaque
%"class.llvm::MachineRegisterInfo" = type { ptr, i8, i8, %"class.llvm::IndexedMap", %"class.llvm::IndexedMap.146", ptr, %"class.llvm::BitVector", %"class.llvm::BitVector", %"class.llvm::BitVector", %"class.std::vector.147", %"class.std::vector.123" }
%"class.llvm::TargetRegisterInfo" = type { ptr, %"class.llvm::MCRegisterInfo", ptr, ptr, ptr, ptr, ptr }
%"struct.llvm::TargetRegisterInfoDesc" = type { i32, i8 }
%"class.llvm::TargetRegisterClass" = type { ptr, ptr, ptr, ptr, ptr, ptr }
%"class.llvm::ArrayRef" = type { ptr, i64 }
%"class.llvm::IndexedMap" = type { %"class.std::vector.140", %"struct.std::pair.145", %"struct.llvm::VirtReg2IndexFunctor" }
%"class.std::vector.140" = type { %"struct.std::_Vector_base.141" }
%"struct.std::_Vector_base.141" = type { %"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *>, std::allocator<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *>, std::allocator<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *> > >::_Vector_impl" = type { ptr, ptr, ptr }
%"struct.std::pair.145" = type { ptr, ptr }
%"class.llvm::MachineOperand" = type { i8, [3 x i8], %union.anon, ptr, %union.anon.188 }
%union.anon = type { i32 }
%union.anon.188 = type { %struct.anon }
%struct.anon = type { ptr, ptr }
%"struct.llvm::VirtReg2IndexFunctor" = type { i8 }
%"class.llvm::IndexedMap.146" = type { %"class.std::vector.147", %"struct.std::pair.152", %"struct.llvm::VirtReg2IndexFunctor" }
%"class.std::vector.147" = type { %"struct.std::_Vector_base.148" }
%"struct.std::_Vector_base.148" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>, std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>, std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" = type { ptr, ptr, ptr }
%"struct.std::pair.152" = type { i32, i32 }
%"class.llvm::BitVector" = type { ptr, i32, i32 }
%"struct.llvm::MachineFunctionInfo" = type { ptr }
%"class.llvm::MachineFrameInfo" = type opaque
%"class.llvm::MachineConstantPool" = type { ptr, i32, %"class.std::vector.153", %"class.llvm::DenseSet" }
%"class.llvm::DataLayout" = type opaque
%"class.std::vector.153" = type { %"struct.std::_Vector_base.154" }
%"struct.std::_Vector_base.154" = type { %"struct.std::_Vector_base<llvm::MachineConstantPoolEntry, std::allocator<llvm::MachineConstantPoolEntry> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MachineConstantPoolEntry, std::allocator<llvm::MachineConstantPoolEntry> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::MachineConstantPoolEntry" = type { %union.anon.158, i32 }
%union.anon.158 = type { ptr }
%"class.llvm::Constant" = type { %"class.llvm::User" }
%"class.llvm::DenseSet" = type { %"class.llvm::DenseMap.159" }
%"class.llvm::DenseMap.159" = type { ptr, i32, i32, i32 }
%"struct.std::pair.162" = type { ptr, i8 }
%"class.llvm::MachineConstantPoolValue" = type { ptr, ptr }
%"class.llvm::MachineJumpTableInfo" = type opaque
%"class.std::vector.163" = type { %"struct.std::_Vector_base.164" }
%"struct.std::_Vector_base.164" = type { %"struct.std::_Vector_base<llvm::MachineBasicBlock *, std::allocator<llvm::MachineBasicBlock *> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MachineBasicBlock *, std::allocator<llvm::MachineBasicBlock *> >::_Vector_impl" = type { ptr, ptr, ptr }
%"class.llvm::Recycler" = type { %"class.llvm::iplist.168" }
%"class.llvm::iplist.168" = type { %"struct.llvm::ilist_traits.169", ptr }
%"struct.llvm::ilist_traits.169" = type { %"struct.llvm::RecyclerStruct" }
%"struct.llvm::RecyclerStruct" = type { ptr, ptr }
%"class.llvm::ArrayRecycler" = type { %"class.llvm::SmallVector.174" }
%"class.llvm::SmallVector.174" = type { %"class.llvm::SmallVectorImpl.175", %"struct.llvm::SmallVectorStorage.179" }
%"class.llvm::SmallVectorImpl.175" = type { %"class.llvm::SmallVectorTemplateBase.176" }
%"class.llvm::SmallVectorTemplateBase.176" = type { %"class.llvm::SmallVectorTemplateCommon.177" }
%"class.llvm::SmallVectorTemplateCommon.177" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.178" }
%"struct.llvm::AlignedCharArrayUnion.178" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::SmallVectorStorage.179" = type { [7 x %"struct.llvm::AlignedCharArrayUnion.178"] }
%"class.llvm::Recycler.180" = type { %"class.llvm::iplist.168" }
%"struct.llvm::ilist.181" = type { %"class.llvm::iplist.182" }
%"class.llvm::iplist.182" = type { %"struct.llvm::ilist_traits.183", ptr }
%"struct.llvm::ilist_traits.183" = type { %"class.llvm::ilist_half_node.1" }
%"class.llvm::ArrayRecycler<llvm::MachineOperand, 8>::Capacity" = type { i8 }
%"class.llvm::ConstantInt" = type { %"class.llvm::Constant", %"class.llvm::APInt" }
%"class.llvm::APInt" = type { i32, %union.anon.189 }
%union.anon.189 = type { i64 }
%"class.llvm::ConstantFP" = type { %"class.llvm::Constant", %"class.llvm::APFloat" }
%"class.llvm::APFloat" = type { ptr, %"union.llvm::APFloat::Significand", i16, i8 }
%"struct.llvm::fltSemantics" = type opaque
%"union.llvm::APFloat::Significand" = type { i64 }
%"class.llvm::BlockAddress" = type { %"class.llvm::Constant" }
%"class.llvm::hash_code" = type { i64 }
%"struct.llvm::hashing::detail::hash_combine_recursive_helper" = type { [64 x i8], %"struct.llvm::hashing::detail::hash_state", i64 }
%"struct.llvm::hashing::detail::hash_state" = type { i64, i64, i64, i64, i64, i64, i64, i64 }
%"class.llvm::PrintReg" = type { ptr, i32, i32 }
%"class.llvm::PseudoSourceValue" = type { %"class.llvm::Value" }
%"class.llvm::FoldingSetNodeID" = type { %"class.llvm::SmallVector.194" }
%"class.llvm::SmallVector.194" = type { [28 x i8], %"struct.llvm::SmallVectorStorage.200" }
%"struct.llvm::SmallVectorStorage.200" = type { [31 x %"struct.llvm::AlignedCharArrayUnion.198"] }
%"struct.llvm::ArrayRecycler<llvm::MachineOperand, 8>::FreeList" = type { ptr }
%"class.llvm::ilist_iterator.202" = type { ptr }
%"class.llvm::TargetInstrInfo" = type { ptr, [28 x i8], i32, i32 }
%"struct.std::pair.203" = type { i8, i8 }
%"class.llvm::SmallVectorImpl.195" = type { %"class.llvm::SmallVectorTemplateBase.196" }
%"class.llvm::SmallVectorTemplateBase.196" = type { %"class.llvm::SmallVectorTemplateCommon.197" }
%"class.llvm::SmallVectorTemplateCommon.197" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.198" }
%"class.llvm::AliasAnalysis" = type { ptr, ptr, ptr, ptr }
%"class.llvm::TargetLibraryInfo" = type opaque
%"struct.llvm::AliasAnalysis::Location" = type { ptr, i64, ptr }
%"class.llvm::DIVariable" = type { %"class.llvm::DIDescriptor" }
%"class.llvm::DIDescriptor" = type { ptr }
%"class.llvm::DIScope" = type { %"class.llvm::DIDescriptor" }
%"class.llvm::ArrayRef.208" = type { ptr, i64 }
%"class.llvm::SmallVector.209" = type { %"class.llvm::SmallVectorImpl.210", %"struct.llvm::SmallVectorStorage.214" }
%"class.llvm::SmallVectorImpl.210" = type { %"class.llvm::SmallVectorTemplateBase.211" }
%"class.llvm::SmallVectorTemplateBase.211" = type { %"class.llvm::SmallVectorTemplateCommon.212" }
%"class.llvm::SmallVectorTemplateCommon.212" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.213" }
%"struct.llvm::AlignedCharArrayUnion.213" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::SmallVectorStorage.214" = type { [7 x %"struct.llvm::AlignedCharArrayUnion.213"] }
%"class.llvm::Twine" = type { %"union.llvm::Twine::Child", %"union.llvm::Twine::Child", i8, i8 }
%"union.llvm::Twine::Child" = type { ptr }
%"struct.std::random_access_iterator_tag" = type { i8 }

declare void @_ZN4llvm19MachineRegisterInfo27removeRegOperandFromUseListEPNS_14MachineOperandE(ptr, ptr)

declare void @_ZN4llvm19MachineRegisterInfo22addRegOperandToUseListEPNS_14MachineOperandE(ptr, ptr)

declare zeroext i32 @_ZNK4llvm14MCRegisterInfo9getSubRegEjj(ptr, i32 zeroext, i32 zeroext)

define void @_ZN4llvm14MachineOperand12substPhysRegEjRKNS_18TargetRegisterInfoE(ptr %this, i32 zeroext %Reg, ptr %TRI) align 2 {
entry:
  %SubReg_TargetFlags.i = getelementptr inbounds %"class.llvm::MachineOperand", ptr %this, i64 0, i32 1
  %bf.load.i = load i24, ptr %SubReg_TargetFlags.i, align 1
  %bf.lshr.i = lshr i24 %bf.load.i, 12
  %tobool = icmp eq i24 %bf.lshr.i, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %bf.cast.i = zext i24 %bf.lshr.i to i32
  %add.ptr = getelementptr inbounds %"class.llvm::TargetRegisterInfo", ptr %TRI, i64 0, i32 1
  %call3 = tail call zeroext i32 @_ZNK4llvm14MCRegisterInfo9getSubRegEjj(ptr %add.ptr, i32 zeroext %Reg, i32 zeroext %bf.cast.i)
  %bf.load.i10 = load i24, ptr %SubReg_TargetFlags.i, align 1
  %bf.clear.i = and i24 %bf.load.i10, 4095
  store i24 %bf.clear.i, ptr %SubReg_TargetFlags.i, align 1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %Reg.addr.0 = phi i32 [ %call3, %if.then ], [ %Reg, %entry ]
  %RegNo.i.i = getelementptr inbounds %"class.llvm::MachineOperand", ptr %this, i64 0, i32 2, i32 0
  %0 = load i32, ptr %RegNo.i.i, align 4
  %cmp.i = icmp eq i32 %0, %Reg.addr.0
  br i1 %cmp.i, label %_ZN4llvm14MachineOperand6setRegEj.exit, label %if.end.i

if.end.i:                                         ; preds = %if.end
  %ParentMI.i.i = getelementptr inbounds %"class.llvm::MachineOperand", ptr %this, i64 0, i32 3
  %1 = load ptr, ptr %ParentMI.i.i, align 8
  %tobool.i = icmp eq ptr %1, null
  br i1 %tobool.i, label %if.end13.i, label %if.then3.i

if.then3.i:                                       ; preds = %if.end.i
  %Parent.i.i = getelementptr inbounds %"class.llvm::MachineInstr", ptr %1, i64 0, i32 2
  %2 = load ptr, ptr %Parent.i.i, align 8
  %tobool5.i = icmp eq ptr %2, null
  br i1 %tobool5.i, label %if.end13.i, label %if.then6.i

if.then6.i:                                       ; preds = %if.then3.i
  %xParent.i.i = getelementptr inbounds %"class.llvm::MachineBasicBlock", ptr %2, i64 0, i32 4
  %3 = load ptr, ptr %xParent.i.i, align 8
  %tobool8.i = icmp eq ptr %3, null
  br i1 %tobool8.i, label %if.end13.i, label %if.then9.i

if.then9.i:                                       ; preds = %if.then6.i
  %RegInfo.i.i = getelementptr inbounds %"class.llvm::MachineFunction", ptr %3, i64 0, i32 5
  %4 = load ptr, ptr %RegInfo.i.i, align 8
  tail call void @_ZN4llvm19MachineRegisterInfo27removeRegOperandFromUseListEPNS_14MachineOperandE(ptr %4, ptr %this)
  store i32 %Reg.addr.0, ptr %RegNo.i.i, align 4
  tail call void @_ZN4llvm19MachineRegisterInfo22addRegOperandToUseListEPNS_14MachineOperandE(ptr %4, ptr %this)
  br label %_ZN4llvm14MachineOperand6setRegEj.exit

if.end13.i:                                       ; preds = %if.then6.i, %if.then3.i, %if.end.i
  store i32 %Reg.addr.0, ptr %RegNo.i.i, align 4
  br label %_ZN4llvm14MachineOperand6setRegEj.exit

_ZN4llvm14MachineOperand6setRegEj.exit:           ; preds = %if.end, %if.then9.i, %if.end13.i
  ret void
}

; CHECK-NOT: lbzu 3, 1(3)
