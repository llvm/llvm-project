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

%"class.llvm::MachineMemOperand" = type { %"struct.llvm::MachinePointerInfo", i64, i32, %"class.llvm::MDNode"*, %"class.llvm::MDNode"* }
%"struct.llvm::MachinePointerInfo" = type { %"class.llvm::Value"*, i64 }
%"class.llvm::Value" = type { i32 (...)**, i8, i8, i16, %"class.llvm::Type"*, %"class.llvm::Use"*, %"class.llvm::StringMapEntry"* }
%"class.llvm::Type" = type { %"class.llvm::LLVMContext"*, i32, i32, %"class.llvm::Type"** }
%"class.llvm::LLVMContext" = type { %"class.llvm::LLVMContextImpl"* }
%"class.llvm::LLVMContextImpl" = type opaque
%"class.llvm::Use" = type { %"class.llvm::Value"*, %"class.llvm::Use"*, %"class.llvm::PointerIntPair" }
%"class.llvm::PointerIntPair" = type { i64 }
%"class.llvm::StringMapEntry" = type opaque
%"class.llvm::MDNode" = type { %"class.llvm::Value", %"class.llvm::FoldingSetImpl::Node", i32, i32 }
%"class.llvm::FoldingSetImpl::Node" = type { i8* }
%"class.llvm::MachineInstr" = type { %"class.llvm::ilist_node", %"class.llvm::MCInstrDesc"*, %"class.llvm::MachineBasicBlock"*, %"class.llvm::MachineOperand"*, i32, %"class.llvm::ArrayRecycler<llvm::MachineOperand, 8>::Capacity", i8, i8, i8, %"class.llvm::MachineMemOperand"**, %"class.llvm::DebugLoc" }
%"class.llvm::ilist_node" = type { %"class.llvm::ilist_half_node", %"class.llvm::MachineInstr"* }
%"class.llvm::ilist_half_node" = type { %"class.llvm::MachineInstr"* }
%"class.llvm::MCInstrDesc" = type { i16, i16, i16, i16, i16, i32, i64, i16*, i16*, %"class.llvm::MCOperandInfo"* }
%"class.llvm::MCOperandInfo" = type { i16, i8, i8, i32 }
%"class.llvm::MachineBasicBlock" = type { %"class.llvm::ilist_node.0", %"struct.llvm::ilist", %"class.llvm::BasicBlock"*, i32, %"class.llvm::MachineFunction"*, %"class.std::vector.163", %"class.std::vector.163", %"class.std::vector.123", %"class.std::vector.123", i32, i8, i8 }
%"class.llvm::ilist_node.0" = type { %"class.llvm::ilist_half_node.1", %"class.llvm::MachineBasicBlock"* }
%"class.llvm::ilist_half_node.1" = type { %"class.llvm::MachineBasicBlock"* }
%"struct.llvm::ilist" = type { %"class.llvm::iplist" }
%"class.llvm::iplist" = type { %"struct.llvm::ilist_traits", %"class.llvm::MachineInstr"* }
%"struct.llvm::ilist_traits" = type { %"class.llvm::ilist_half_node", %"class.llvm::MachineBasicBlock"* }
%"class.llvm::BasicBlock" = type { %"class.llvm::Value", %"class.llvm::ilist_node.2", %"class.llvm::iplist.4", %"class.llvm::Function"* }
%"class.llvm::ilist_node.2" = type { %"class.llvm::ilist_half_node.3", %"class.llvm::BasicBlock"* }
%"class.llvm::ilist_half_node.3" = type { %"class.llvm::BasicBlock"* }
%"class.llvm::iplist.4" = type { %"struct.llvm::ilist_traits.5", %"class.llvm::Instruction"* }
%"struct.llvm::ilist_traits.5" = type { %"class.llvm::ilist_half_node.10" }
%"class.llvm::ilist_half_node.10" = type { %"class.llvm::Instruction"* }
%"class.llvm::Instruction" = type { %"class.llvm::User", %"class.llvm::ilist_node.193", %"class.llvm::BasicBlock"*, %"class.llvm::DebugLoc" }
%"class.llvm::User" = type { %"class.llvm::Value", %"class.llvm::Use"*, i32 }
%"class.llvm::ilist_node.193" = type { %"class.llvm::ilist_half_node.10", %"class.llvm::Instruction"* }
%"class.llvm::DebugLoc" = type { i32, i32 }
%"class.llvm::Function" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.27", %"class.llvm::iplist.47", %"class.llvm::iplist.54", %"class.llvm::ValueSymbolTable"*, %"class.llvm::AttributeSet" }
%"class.llvm::GlobalValue" = type { [52 x i8], [4 x i8], %"class.llvm::Module"*, %"class.std::basic_string" }
%"class.llvm::Module" = type { %"class.llvm::LLVMContext"*, %"class.llvm::iplist.11", %"class.llvm::iplist.20", %"class.llvm::iplist.29", %"struct.llvm::ilist.38", %"class.std::basic_string", %"class.llvm::ValueSymbolTable"*, %"class.llvm::OwningPtr", %"class.std::basic_string", %"class.std::basic_string", %"class.std::basic_string", i8* }
%"class.llvm::iplist.11" = type { %"struct.llvm::ilist_traits.12", %"class.llvm::GlobalVariable"* }
%"struct.llvm::ilist_traits.12" = type { %"class.llvm::ilist_node.18" }
%"class.llvm::ilist_node.18" = type { %"class.llvm::ilist_half_node.19", %"class.llvm::GlobalVariable"* }
%"class.llvm::ilist_half_node.19" = type { %"class.llvm::GlobalVariable"* }
%"class.llvm::GlobalVariable" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.18", i8 }
%"class.llvm::iplist.20" = type { %"struct.llvm::ilist_traits.21", %"class.llvm::Function"* }
%"struct.llvm::ilist_traits.21" = type { %"class.llvm::ilist_node.27" }
%"class.llvm::ilist_node.27" = type { %"class.llvm::ilist_half_node.28", %"class.llvm::Function"* }
%"class.llvm::ilist_half_node.28" = type { %"class.llvm::Function"* }
%"class.llvm::iplist.29" = type { %"struct.llvm::ilist_traits.30", %"class.llvm::GlobalAlias"* }
%"struct.llvm::ilist_traits.30" = type { %"class.llvm::ilist_node.36" }
%"class.llvm::ilist_node.36" = type { %"class.llvm::ilist_half_node.37", %"class.llvm::GlobalAlias"* }
%"class.llvm::ilist_half_node.37" = type { %"class.llvm::GlobalAlias"* }
%"class.llvm::GlobalAlias" = type { %"class.llvm::GlobalValue", %"class.llvm::ilist_node.36" }
%"struct.llvm::ilist.38" = type { %"class.llvm::iplist.39" }
%"class.llvm::iplist.39" = type { %"struct.llvm::ilist_traits.40", %"class.llvm::NamedMDNode"* }
%"struct.llvm::ilist_traits.40" = type { %"class.llvm::ilist_node.45" }
%"class.llvm::ilist_node.45" = type { %"class.llvm::ilist_half_node.46", %"class.llvm::NamedMDNode"* }
%"class.llvm::ilist_half_node.46" = type { %"class.llvm::NamedMDNode"* }
%"class.llvm::NamedMDNode" = type { %"class.llvm::ilist_node.45", %"class.std::basic_string", %"class.llvm::Module"*, i8* }
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%"class.llvm::ValueSymbolTable" = type opaque
%"class.llvm::OwningPtr" = type { %"class.llvm::GVMaterializer"* }
%"class.llvm::GVMaterializer" = type opaque
%"class.llvm::iplist.47" = type { %"struct.llvm::ilist_traits.48", %"class.llvm::BasicBlock"* }
%"struct.llvm::ilist_traits.48" = type { %"class.llvm::ilist_half_node.3" }
%"class.llvm::iplist.54" = type { %"struct.llvm::ilist_traits.55", %"class.llvm::Argument"* }
%"struct.llvm::ilist_traits.55" = type { %"class.llvm::ilist_half_node.61" }
%"class.llvm::ilist_half_node.61" = type { %"class.llvm::Argument"* }
%"class.llvm::Argument" = type { %"class.llvm::Value", %"class.llvm::ilist_node.192", %"class.llvm::Function"* }
%"class.llvm::ilist_node.192" = type { %"class.llvm::ilist_half_node.61", %"class.llvm::Argument"* }
%"class.llvm::AttributeSet" = type { %"class.llvm::AttributeSetImpl"* }
%"class.llvm::AttributeSetImpl" = type opaque
%"class.llvm::MachineFunction" = type { %"class.llvm::Function"*, %"class.llvm::TargetMachine"*, %"class.llvm::MCContext"*, %"class.llvm::MachineModuleInfo"*, %"class.llvm::GCModuleInfo"*, %"class.llvm::MachineRegisterInfo"*, %"struct.llvm::MachineFunctionInfo"*, %"class.llvm::MachineFrameInfo"*, %"class.llvm::MachineConstantPool"*, %"class.llvm::MachineJumpTableInfo"*, %"class.std::vector.163", %"class.llvm::BumpPtrAllocator", %"class.llvm::Recycler", %"class.llvm::ArrayRecycler", %"class.llvm::Recycler.180", %"struct.llvm::ilist.181", i32, i32, i8 }
%"class.llvm::TargetMachine" = type { i32 (...)**, %"class.llvm::Target"*, %"class.std::basic_string", %"class.std::basic_string", %"class.std::basic_string", %"class.llvm::MCCodeGenInfo"*, %"class.llvm::MCAsmInfo"*, i8, %"class.llvm::TargetOptions" }
%"class.llvm::Target" = type opaque
%"class.llvm::MCCodeGenInfo" = type opaque
%"class.llvm::MCAsmInfo" = type opaque
%"class.llvm::TargetOptions" = type { [2 x i8], i32, i8, i32, i8, %"class.std::basic_string", i32, i32 }
%"class.llvm::MCContext" = type { %"class.llvm::SourceMgr"*, %"class.llvm::MCAsmInfo"*, %"class.llvm::MCRegisterInfo"*, %"class.llvm::MCObjectFileInfo"*, %"class.llvm::BumpPtrAllocator", %"class.llvm::StringMap", %"class.llvm::StringMap.62", i32, %"class.llvm::DenseMap.63", i8*, %"class.llvm::raw_ostream"*, i8, %"class.std::basic_string", %"class.std::basic_string", %"class.std::vector", %"class.std::vector.70", %"class.llvm::MCDwarfLoc", i8, i8, i32, %"class.llvm::MCSection"*, %"class.llvm::MCSymbol"*, %"class.llvm::MCSymbol"*, %"class.std::vector.75", %"class.llvm::StringRef", %"class.llvm::StringRef", i8, %"class.llvm::DenseMap.80", %"class.std::vector.84", i8*, i8*, i8*, i8 }
%"class.llvm::SourceMgr" = type opaque
%"class.llvm::MCRegisterInfo" = type { %"struct.llvm::MCRegisterDesc"*, i32, i32, i32, %"class.llvm::MCRegisterClass"*, i32, i32, [2 x i16]*, i16*, i8*, i16*, i32, i16*, i32, i32, i32, i32, %"struct.llvm::MCRegisterInfo::DwarfLLVMRegPair"*, %"struct.llvm::MCRegisterInfo::DwarfLLVMRegPair"*, %"struct.llvm::MCRegisterInfo::DwarfLLVMRegPair"*, %"struct.llvm::MCRegisterInfo::DwarfLLVMRegPair"*, %"class.llvm::DenseMap" }
%"struct.llvm::MCRegisterDesc" = type { i32, i32, i32, i32, i32, i32 }
%"class.llvm::MCRegisterClass" = type { i8*, i16*, i8*, i16, i16, i16, i16, i16, i8, i8 }
%"struct.llvm::MCRegisterInfo::DwarfLLVMRegPair" = type { i32, i32 }
%"class.llvm::DenseMap" = type { %"struct.std::pair"*, i32, i32, i32 }
%"struct.std::pair" = type { i32, i32 }
%"class.llvm::MCObjectFileInfo" = type opaque
%"class.llvm::BumpPtrAllocator" = type { i64, i64, %"class.llvm::SlabAllocator"*, %"class.llvm::MemSlab"*, i8*, i8*, i64 }
%"class.llvm::SlabAllocator" = type { i32 (...)** }
%"class.llvm::MemSlab" = type { i64, %"class.llvm::MemSlab"* }
%"class.llvm::StringMap" = type { %"class.llvm::StringMapImpl", %"class.llvm::BumpPtrAllocator"* }
%"class.llvm::StringMapImpl" = type { %"class.llvm::StringMapEntryBase"**, i32, i32, i32, i32 }
%"class.llvm::StringMapEntryBase" = type { i32 }
%"class.llvm::StringMap.62" = type { %"class.llvm::StringMapImpl", %"class.llvm::BumpPtrAllocator"* }
%"class.llvm::DenseMap.63" = type { %"struct.std::pair.66"*, i32, i32, i32 }
%"struct.std::pair.66" = type opaque
%"class.llvm::raw_ostream" = type { i32 (...)**, i8*, i8*, i8*, i32 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<llvm::MCDwarfFile *, std::allocator<llvm::MCDwarfFile *> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MCDwarfFile *, std::allocator<llvm::MCDwarfFile *> >::_Vector_impl" = type { %"class.llvm::MCDwarfFile"**, %"class.llvm::MCDwarfFile"**, %"class.llvm::MCDwarfFile"** }
%"class.llvm::MCDwarfFile" = type { %"class.llvm::StringRef", i32 }
%"class.llvm::StringRef" = type { i8*, i64 }
%"class.std::vector.70" = type { %"struct.std::_Vector_base.71" }
%"struct.std::_Vector_base.71" = type { %"struct.std::_Vector_base<llvm::StringRef, std::allocator<llvm::StringRef> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::StringRef, std::allocator<llvm::StringRef> >::_Vector_impl" = type { %"class.llvm::StringRef"*, %"class.llvm::StringRef"*, %"class.llvm::StringRef"* }
%"class.llvm::MCDwarfLoc" = type { i32, i32, i32, i32, i32, i32 }
%"class.llvm::MCSection" = type opaque
%"class.llvm::MCSymbol" = type { %"class.llvm::StringRef", %"class.llvm::MCSection"*, %"class.llvm::MCExpr"*, i8 }
%"class.llvm::MCExpr" = type opaque
%"class.std::vector.75" = type { %"struct.std::_Vector_base.76" }
%"struct.std::_Vector_base.76" = type { %"struct.std::_Vector_base<const llvm::MCGenDwarfLabelEntry *, std::allocator<const llvm::MCGenDwarfLabelEntry *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::MCGenDwarfLabelEntry *, std::allocator<const llvm::MCGenDwarfLabelEntry *> >::_Vector_impl" = type { %"class.llvm::MCGenDwarfLabelEntry"**, %"class.llvm::MCGenDwarfLabelEntry"**, %"class.llvm::MCGenDwarfLabelEntry"** }
%"class.llvm::MCGenDwarfLabelEntry" = type { %"class.llvm::StringRef", i32, i32, %"class.llvm::MCSymbol"* }
%"class.llvm::DenseMap.80" = type { %"struct.std::pair.83"*, i32, i32, i32 }
%"struct.std::pair.83" = type { %"class.llvm::MCSection"*, %"class.llvm::MCLineSection"* }
%"class.llvm::MCLineSection" = type { %"class.std::vector.215" }
%"class.std::vector.215" = type { %"struct.std::_Vector_base.216" }
%"struct.std::_Vector_base.216" = type { %"struct.std::_Vector_base<llvm::MCLineEntry, std::allocator<llvm::MCLineEntry> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MCLineEntry, std::allocator<llvm::MCLineEntry> >::_Vector_impl" = type { %"class.llvm::MCLineEntry"*, %"class.llvm::MCLineEntry"*, %"class.llvm::MCLineEntry"* }
%"class.llvm::MCLineEntry" = type { %"class.llvm::MCDwarfLoc", %"class.llvm::MCSymbol"* }
%"class.std::vector.84" = type { %"struct.std::_Vector_base.85" }
%"struct.std::_Vector_base.85" = type { %"struct.std::_Vector_base<const llvm::MCSection *, std::allocator<const llvm::MCSection *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::MCSection *, std::allocator<const llvm::MCSection *> >::_Vector_impl" = type { %"class.llvm::MCSection"**, %"class.llvm::MCSection"**, %"class.llvm::MCSection"** }
%"class.llvm::MachineModuleInfo" = type { %"class.llvm::ImmutablePass", %"class.llvm::MCContext", %"class.llvm::Module"*, %"class.llvm::MachineModuleInfoImpl"*, %"class.std::vector.95", i32, %"class.std::vector.100", %"class.llvm::DenseMap.110", %"class.llvm::DenseMap.114", i32, %"class.std::vector.118", %"class.std::vector.123", %"class.std::vector.123", %"class.std::vector.128", %"class.llvm::SmallPtrSet", %"class.llvm::MMIAddrLabelMap"*, i8, i8, i8, i8, %"class.llvm::SmallVector.133" }
%"class.llvm::ImmutablePass" = type { %"class.llvm::ModulePass" }
%"class.llvm::ModulePass" = type { %"class.llvm::Pass" }
%"class.llvm::Pass" = type { i32 (...)**, %"class.llvm::AnalysisResolver"*, i8*, i32 }
%"class.llvm::AnalysisResolver" = type { %"class.std::vector.89", %"class.llvm::PMDataManager"* }
%"class.std::vector.89" = type { %"struct.std::_Vector_base.90" }
%"struct.std::_Vector_base.90" = type { %"struct.std::_Vector_base<std::pair<const void *, llvm::Pass *>, std::allocator<std::pair<const void *, llvm::Pass *> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<const void *, llvm::Pass *>, std::allocator<std::pair<const void *, llvm::Pass *> > >::_Vector_impl" = type { %"struct.std::pair.94"*, %"struct.std::pair.94"*, %"struct.std::pair.94"* }
%"struct.std::pair.94" = type { i8*, %"class.llvm::Pass"* }
%"class.llvm::PMDataManager" = type opaque
%"class.llvm::MachineModuleInfoImpl" = type { i32 (...)** }
%"class.std::vector.95" = type { %"struct.std::_Vector_base.96" }
%"struct.std::_Vector_base.96" = type { %"struct.std::_Vector_base<llvm::MachineMove, std::allocator<llvm::MachineMove> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MachineMove, std::allocator<llvm::MachineMove> >::_Vector_impl" = type { %"class.llvm::MachineMove"*, %"class.llvm::MachineMove"*, %"class.llvm::MachineMove"* }
%"class.llvm::MachineMove" = type { %"class.llvm::MCSymbol"*, %"class.llvm::MachineLocation", %"class.llvm::MachineLocation" }
%"class.llvm::MachineLocation" = type { i8, i32, i32 }
%"class.std::vector.100" = type { %"struct.std::_Vector_base.101" }
%"struct.std::_Vector_base.101" = type { %"struct.std::_Vector_base<llvm::LandingPadInfo, std::allocator<llvm::LandingPadInfo> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::LandingPadInfo, std::allocator<llvm::LandingPadInfo> >::_Vector_impl" = type { %"struct.llvm::LandingPadInfo"*, %"struct.llvm::LandingPadInfo"*, %"struct.llvm::LandingPadInfo"* }
%"struct.llvm::LandingPadInfo" = type { %"class.llvm::MachineBasicBlock"*, %"class.llvm::SmallVector", %"class.llvm::SmallVector", %"class.llvm::MCSymbol"*, %"class.llvm::Function"*, %"class.std::vector.105" }
%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl", %"struct.llvm::SmallVectorStorage" }
%"class.llvm::SmallVectorImpl" = type { %"class.llvm::SmallVectorTemplateBase" }
%"class.llvm::SmallVectorTemplateBase" = type { %"class.llvm::SmallVectorTemplateCommon" }
%"class.llvm::SmallVectorTemplateCommon" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion" }
%"class.llvm::SmallVectorBase" = type { i8*, i8*, i8* }
%"struct.llvm::AlignedCharArrayUnion" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::AlignedCharArray" = type { [8 x i8] }
%"struct.llvm::SmallVectorStorage" = type { i8 }
%"class.std::vector.105" = type { %"struct.std::_Vector_base.106" }
%"struct.std::_Vector_base.106" = type { %"struct.std::_Vector_base<int, std::allocator<int> >::_Vector_impl" }
%"struct.std::_Vector_base<int, std::allocator<int> >::_Vector_impl" = type { i32*, i32*, i32* }
%"class.llvm::DenseMap.110" = type { %"struct.std::pair.113"*, i32, i32, i32 }
%"struct.std::pair.113" = type { %"class.llvm::MCSymbol"*, %"class.llvm::SmallVector.206" }
%"class.llvm::SmallVector.206" = type { [28 x i8], %"struct.llvm::SmallVectorStorage.207" }
%"struct.llvm::SmallVectorStorage.207" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.198"] }
%"struct.llvm::AlignedCharArrayUnion.198" = type { %"struct.llvm::AlignedCharArray.199" }
%"struct.llvm::AlignedCharArray.199" = type { [4 x i8] }
%"class.llvm::DenseMap.114" = type { %"struct.std::pair.117"*, i32, i32, i32 }
%"struct.std::pair.117" = type { %"class.llvm::MCSymbol"*, i32 }
%"class.std::vector.118" = type { %"struct.std::_Vector_base.119" }
%"struct.std::_Vector_base.119" = type { %"struct.std::_Vector_base<const llvm::GlobalVariable *, std::allocator<const llvm::GlobalVariable *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::GlobalVariable *, std::allocator<const llvm::GlobalVariable *> >::_Vector_impl" = type { %"class.llvm::GlobalVariable"**, %"class.llvm::GlobalVariable"**, %"class.llvm::GlobalVariable"** }
%"class.std::vector.123" = type { %"struct.std::_Vector_base.124" }
%"struct.std::_Vector_base.124" = type { %"struct.std::_Vector_base<unsigned int, std::allocator<unsigned int> >::_Vector_impl" }
%"struct.std::_Vector_base<unsigned int, std::allocator<unsigned int> >::_Vector_impl" = type { i32*, i32*, i32* }
%"class.std::vector.128" = type { %"struct.std::_Vector_base.129" }
%"struct.std::_Vector_base.129" = type { %"struct.std::_Vector_base<const llvm::Function *, std::allocator<const llvm::Function *> >::_Vector_impl" }
%"struct.std::_Vector_base<const llvm::Function *, std::allocator<const llvm::Function *> >::_Vector_impl" = type { %"class.llvm::Function"**, %"class.llvm::Function"**, %"class.llvm::Function"** }
%"class.llvm::SmallPtrSet" = type { %"class.llvm::SmallPtrSetImpl", [33 x i8*] }
%"class.llvm::SmallPtrSetImpl" = type { i8**, i8**, i32, i32, i32 }
%"class.llvm::MMIAddrLabelMap" = type opaque
%"class.llvm::SmallVector.133" = type { %"class.llvm::SmallVectorImpl.134", %"struct.llvm::SmallVectorStorage.139" }
%"class.llvm::SmallVectorImpl.134" = type { %"class.llvm::SmallVectorTemplateBase.135" }
%"class.llvm::SmallVectorTemplateBase.135" = type { %"class.llvm::SmallVectorTemplateCommon.136" }
%"class.llvm::SmallVectorTemplateCommon.136" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.137" }
%"struct.llvm::AlignedCharArrayUnion.137" = type { %"struct.llvm::AlignedCharArray.138" }
%"struct.llvm::AlignedCharArray.138" = type { [40 x i8] }
%"struct.llvm::SmallVectorStorage.139" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.137"] }
%"class.llvm::GCModuleInfo" = type opaque
%"class.llvm::MachineRegisterInfo" = type { %"class.llvm::TargetRegisterInfo"*, i8, i8, %"class.llvm::IndexedMap", %"class.llvm::IndexedMap.146", %"class.llvm::MachineOperand"**, %"class.llvm::BitVector", %"class.llvm::BitVector", %"class.llvm::BitVector", %"class.std::vector.147", %"class.std::vector.123" }
%"class.llvm::TargetRegisterInfo" = type { i32 (...)**, %"class.llvm::MCRegisterInfo", %"struct.llvm::TargetRegisterInfoDesc"*, i8**, i32*, %"class.llvm::TargetRegisterClass"**, %"class.llvm::TargetRegisterClass"** }
%"struct.llvm::TargetRegisterInfoDesc" = type { i32, i8 }
%"class.llvm::TargetRegisterClass" = type { %"class.llvm::MCRegisterClass"*, i32*, i32*, i16*, %"class.llvm::TargetRegisterClass"**, void (%"class.llvm::ArrayRef"*, %"class.llvm::MachineFunction"*)* }
%"class.llvm::ArrayRef" = type { i16*, i64 }
%"class.llvm::IndexedMap" = type { %"class.std::vector.140", %"struct.std::pair.145", %"struct.llvm::VirtReg2IndexFunctor" }
%"class.std::vector.140" = type { %"struct.std::_Vector_base.141" }
%"struct.std::_Vector_base.141" = type { %"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *>, std::allocator<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *>, std::allocator<std::pair<const llvm::TargetRegisterClass *, llvm::MachineOperand *> > >::_Vector_impl" = type { %"struct.std::pair.145"*, %"struct.std::pair.145"*, %"struct.std::pair.145"* }
%"struct.std::pair.145" = type { %"class.llvm::TargetRegisterClass"*, %"class.llvm::MachineOperand"* }
%"class.llvm::MachineOperand" = type { i8, [3 x i8], %union.anon, %"class.llvm::MachineInstr"*, %union.anon.188 }
%union.anon = type { i32 }
%union.anon.188 = type { %struct.anon }
%struct.anon = type { %"class.llvm::MachineOperand"*, %"class.llvm::MachineOperand"* }
%"struct.llvm::VirtReg2IndexFunctor" = type { i8 }
%"class.llvm::IndexedMap.146" = type { %"class.std::vector.147", %"struct.std::pair.152", %"struct.llvm::VirtReg2IndexFunctor" }
%"class.std::vector.147" = type { %"struct.std::_Vector_base.148" }
%"struct.std::_Vector_base.148" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>, std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>, std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" = type { %"struct.std::pair.152"*, %"struct.std::pair.152"*, %"struct.std::pair.152"* }
%"struct.std::pair.152" = type { i32, i32 }
%"class.llvm::BitVector" = type { i64*, i32, i32 }
%"struct.llvm::MachineFunctionInfo" = type { i32 (...)** }
%"class.llvm::MachineFrameInfo" = type opaque
%"class.llvm::MachineConstantPool" = type { %"class.llvm::DataLayout"*, i32, %"class.std::vector.153", %"class.llvm::DenseSet" }
%"class.llvm::DataLayout" = type opaque
%"class.std::vector.153" = type { %"struct.std::_Vector_base.154" }
%"struct.std::_Vector_base.154" = type { %"struct.std::_Vector_base<llvm::MachineConstantPoolEntry, std::allocator<llvm::MachineConstantPoolEntry> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MachineConstantPoolEntry, std::allocator<llvm::MachineConstantPoolEntry> >::_Vector_impl" = type { %"class.llvm::MachineConstantPoolEntry"*, %"class.llvm::MachineConstantPoolEntry"*, %"class.llvm::MachineConstantPoolEntry"* }
%"class.llvm::MachineConstantPoolEntry" = type { %union.anon.158, i32 }
%union.anon.158 = type { %"class.llvm::Constant"* }
%"class.llvm::Constant" = type { %"class.llvm::User" }
%"class.llvm::DenseSet" = type { %"class.llvm::DenseMap.159" }
%"class.llvm::DenseMap.159" = type { %"struct.std::pair.162"*, i32, i32, i32 }
%"struct.std::pair.162" = type { %"class.llvm::MachineConstantPoolValue"*, i8 }
%"class.llvm::MachineConstantPoolValue" = type { i32 (...)**, %"class.llvm::Type"* }
%"class.llvm::MachineJumpTableInfo" = type opaque
%"class.std::vector.163" = type { %"struct.std::_Vector_base.164" }
%"struct.std::_Vector_base.164" = type { %"struct.std::_Vector_base<llvm::MachineBasicBlock *, std::allocator<llvm::MachineBasicBlock *> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::MachineBasicBlock *, std::allocator<llvm::MachineBasicBlock *> >::_Vector_impl" = type { %"class.llvm::MachineBasicBlock"**, %"class.llvm::MachineBasicBlock"**, %"class.llvm::MachineBasicBlock"** }
%"class.llvm::Recycler" = type { %"class.llvm::iplist.168" }
%"class.llvm::iplist.168" = type { %"struct.llvm::ilist_traits.169", %"struct.llvm::RecyclerStruct"* }
%"struct.llvm::ilist_traits.169" = type { %"struct.llvm::RecyclerStruct" }
%"struct.llvm::RecyclerStruct" = type { %"struct.llvm::RecyclerStruct"*, %"struct.llvm::RecyclerStruct"* }
%"class.llvm::ArrayRecycler" = type { %"class.llvm::SmallVector.174" }
%"class.llvm::SmallVector.174" = type { %"class.llvm::SmallVectorImpl.175", %"struct.llvm::SmallVectorStorage.179" }
%"class.llvm::SmallVectorImpl.175" = type { %"class.llvm::SmallVectorTemplateBase.176" }
%"class.llvm::SmallVectorTemplateBase.176" = type { %"class.llvm::SmallVectorTemplateCommon.177" }
%"class.llvm::SmallVectorTemplateCommon.177" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.178" }
%"struct.llvm::AlignedCharArrayUnion.178" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::SmallVectorStorage.179" = type { [7 x %"struct.llvm::AlignedCharArrayUnion.178"] }
%"class.llvm::Recycler.180" = type { %"class.llvm::iplist.168" }
%"struct.llvm::ilist.181" = type { %"class.llvm::iplist.182" }
%"class.llvm::iplist.182" = type { %"struct.llvm::ilist_traits.183", %"class.llvm::MachineBasicBlock"* }
%"struct.llvm::ilist_traits.183" = type { %"class.llvm::ilist_half_node.1" }
%"class.llvm::ArrayRecycler<llvm::MachineOperand, 8>::Capacity" = type { i8 }
%"class.llvm::ConstantInt" = type { %"class.llvm::Constant", %"class.llvm::APInt" }
%"class.llvm::APInt" = type { i32, %union.anon.189 }
%union.anon.189 = type { i64 }
%"class.llvm::ConstantFP" = type { %"class.llvm::Constant", %"class.llvm::APFloat" }
%"class.llvm::APFloat" = type { %"struct.llvm::fltSemantics"*, %"union.llvm::APFloat::Significand", i16, i8 }
%"struct.llvm::fltSemantics" = type opaque
%"union.llvm::APFloat::Significand" = type { i64 }
%"class.llvm::BlockAddress" = type { %"class.llvm::Constant" }
%"class.llvm::hash_code" = type { i64 }
%"struct.llvm::hashing::detail::hash_combine_recursive_helper" = type { [64 x i8], %"struct.llvm::hashing::detail::hash_state", i64 }
%"struct.llvm::hashing::detail::hash_state" = type { i64, i64, i64, i64, i64, i64, i64, i64 }
%"class.llvm::PrintReg" = type { %"class.llvm::TargetRegisterInfo"*, i32, i32 }
%"class.llvm::PseudoSourceValue" = type { %"class.llvm::Value" }
%"class.llvm::FoldingSetNodeID" = type { %"class.llvm::SmallVector.194" }
%"class.llvm::SmallVector.194" = type { [28 x i8], %"struct.llvm::SmallVectorStorage.200" }
%"struct.llvm::SmallVectorStorage.200" = type { [31 x %"struct.llvm::AlignedCharArrayUnion.198"] }
%"struct.llvm::ArrayRecycler<llvm::MachineOperand, 8>::FreeList" = type { %"struct.llvm::ArrayRecycler<llvm::MachineOperand, 8>::FreeList"* }
%"class.llvm::ilist_iterator.202" = type { %"class.llvm::MachineInstr"* }
%"class.llvm::TargetInstrInfo" = type { i32 (...)**, [28 x i8], i32, i32 }
%"struct.std::pair.203" = type { i8, i8 }
%"class.llvm::SmallVectorImpl.195" = type { %"class.llvm::SmallVectorTemplateBase.196" }
%"class.llvm::SmallVectorTemplateBase.196" = type { %"class.llvm::SmallVectorTemplateCommon.197" }
%"class.llvm::SmallVectorTemplateCommon.197" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.198" }
%"class.llvm::AliasAnalysis" = type { i32 (...)**, %"class.llvm::DataLayout"*, %"class.llvm::TargetLibraryInfo"*, %"class.llvm::AliasAnalysis"* }
%"class.llvm::TargetLibraryInfo" = type opaque
%"struct.llvm::AliasAnalysis::Location" = type { %"class.llvm::Value"*, i64, %"class.llvm::MDNode"* }
%"class.llvm::DIVariable" = type { %"class.llvm::DIDescriptor" }
%"class.llvm::DIDescriptor" = type { %"class.llvm::MDNode"* }
%"class.llvm::DIScope" = type { %"class.llvm::DIDescriptor" }
%"class.llvm::ArrayRef.208" = type { i32*, i64 }
%"class.llvm::SmallVector.209" = type { %"class.llvm::SmallVectorImpl.210", %"struct.llvm::SmallVectorStorage.214" }
%"class.llvm::SmallVectorImpl.210" = type { %"class.llvm::SmallVectorTemplateBase.211" }
%"class.llvm::SmallVectorTemplateBase.211" = type { %"class.llvm::SmallVectorTemplateCommon.212" }
%"class.llvm::SmallVectorTemplateCommon.212" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.213" }
%"struct.llvm::AlignedCharArrayUnion.213" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::SmallVectorStorage.214" = type { [7 x %"struct.llvm::AlignedCharArrayUnion.213"] }
%"class.llvm::Twine" = type { %"union.llvm::Twine::Child", %"union.llvm::Twine::Child", i8, i8 }
%"union.llvm::Twine::Child" = type { %"class.llvm::Twine"* }
%"struct.std::random_access_iterator_tag" = type { i8 }

declare void @_ZN4llvm19MachineRegisterInfo27removeRegOperandFromUseListEPNS_14MachineOperandE(%"class.llvm::MachineRegisterInfo"*, %"class.llvm::MachineOperand"*)

declare void @_ZN4llvm19MachineRegisterInfo22addRegOperandToUseListEPNS_14MachineOperandE(%"class.llvm::MachineRegisterInfo"*, %"class.llvm::MachineOperand"*)

declare zeroext i32 @_ZNK4llvm14MCRegisterInfo9getSubRegEjj(%"class.llvm::MCRegisterInfo"*, i32 zeroext, i32 zeroext)

define void @_ZN4llvm14MachineOperand12substPhysRegEjRKNS_18TargetRegisterInfoE(%"class.llvm::MachineOperand"* %this, i32 zeroext %Reg, %"class.llvm::TargetRegisterInfo"* %TRI) align 2 {
entry:
  %SubReg_TargetFlags.i = getelementptr inbounds %"class.llvm::MachineOperand", %"class.llvm::MachineOperand"* %this, i64 0, i32 1
  %0 = bitcast [3 x i8]* %SubReg_TargetFlags.i to i24*
  %bf.load.i = load i24, i24* %0, align 1
  %bf.lshr.i = lshr i24 %bf.load.i, 12
  %tobool = icmp eq i24 %bf.lshr.i, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %bf.cast.i = zext i24 %bf.lshr.i to i32
  %add.ptr = getelementptr inbounds %"class.llvm::TargetRegisterInfo", %"class.llvm::TargetRegisterInfo"* %TRI, i64 0, i32 1
  %call3 = tail call zeroext i32 @_ZNK4llvm14MCRegisterInfo9getSubRegEjj(%"class.llvm::MCRegisterInfo"* %add.ptr, i32 zeroext %Reg, i32 zeroext %bf.cast.i)
  %bf.load.i10 = load i24, i24* %0, align 1
  %bf.clear.i = and i24 %bf.load.i10, 4095
  store i24 %bf.clear.i, i24* %0, align 1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %Reg.addr.0 = phi i32 [ %call3, %if.then ], [ %Reg, %entry ]
  %RegNo.i.i = getelementptr inbounds %"class.llvm::MachineOperand", %"class.llvm::MachineOperand"* %this, i64 0, i32 2, i32 0
  %1 = load i32, i32* %RegNo.i.i, align 4
  %cmp.i = icmp eq i32 %1, %Reg.addr.0
  br i1 %cmp.i, label %_ZN4llvm14MachineOperand6setRegEj.exit, label %if.end.i

if.end.i:                                         ; preds = %if.end
  %ParentMI.i.i = getelementptr inbounds %"class.llvm::MachineOperand", %"class.llvm::MachineOperand"* %this, i64 0, i32 3
  %2 = load %"class.llvm::MachineInstr"*, %"class.llvm::MachineInstr"** %ParentMI.i.i, align 8
  %tobool.i = icmp eq %"class.llvm::MachineInstr"* %2, null
  br i1 %tobool.i, label %if.end13.i, label %if.then3.i

if.then3.i:                                       ; preds = %if.end.i
  %Parent.i.i = getelementptr inbounds %"class.llvm::MachineInstr", %"class.llvm::MachineInstr"* %2, i64 0, i32 2
  %3 = load %"class.llvm::MachineBasicBlock"*, %"class.llvm::MachineBasicBlock"** %Parent.i.i, align 8
  %tobool5.i = icmp eq %"class.llvm::MachineBasicBlock"* %3, null
  br i1 %tobool5.i, label %if.end13.i, label %if.then6.i

if.then6.i:                                       ; preds = %if.then3.i
  %xParent.i.i = getelementptr inbounds %"class.llvm::MachineBasicBlock", %"class.llvm::MachineBasicBlock"* %3, i64 0, i32 4
  %4 = load %"class.llvm::MachineFunction"*, %"class.llvm::MachineFunction"** %xParent.i.i, align 8
  %tobool8.i = icmp eq %"class.llvm::MachineFunction"* %4, null
  br i1 %tobool8.i, label %if.end13.i, label %if.then9.i

if.then9.i:                                       ; preds = %if.then6.i
  %RegInfo.i.i = getelementptr inbounds %"class.llvm::MachineFunction", %"class.llvm::MachineFunction"* %4, i64 0, i32 5
  %5 = load %"class.llvm::MachineRegisterInfo"*, %"class.llvm::MachineRegisterInfo"** %RegInfo.i.i, align 8
  tail call void @_ZN4llvm19MachineRegisterInfo27removeRegOperandFromUseListEPNS_14MachineOperandE(%"class.llvm::MachineRegisterInfo"* %5, %"class.llvm::MachineOperand"* %this)
  store i32 %Reg.addr.0, i32* %RegNo.i.i, align 4
  tail call void @_ZN4llvm19MachineRegisterInfo22addRegOperandToUseListEPNS_14MachineOperandE(%"class.llvm::MachineRegisterInfo"* %5, %"class.llvm::MachineOperand"* %this)
  br label %_ZN4llvm14MachineOperand6setRegEj.exit

if.end13.i:                                       ; preds = %if.then6.i, %if.then3.i, %if.end.i
  store i32 %Reg.addr.0, i32* %RegNo.i.i, align 4
  br label %_ZN4llvm14MachineOperand6setRegEj.exit

_ZN4llvm14MachineOperand6setRegEj.exit:           ; preds = %if.end, %if.then9.i, %if.end13.i
  ret void
}

; CHECK-NOT: lbzu 3, 1(3)
