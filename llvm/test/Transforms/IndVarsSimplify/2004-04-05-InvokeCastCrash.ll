; RUN: llvm-as < %s | opt -indvars -disable-output
; ModuleID = '2004-04-05-InvokeCastCrash.ll'
	%struct.__false_type = type { i8 }
	%"struct.__gnu_cxx::_Hashtable_node<const llvm::Constant*>" = type { %"struct.__gnu_cxx::_Hashtable_node<const llvm::Constant*>"*, %"struct.llvm::Constant"* }
	%"struct.__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >" = type { %"struct.__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >"*, %"struct.std::pair<const llvm::Value* const,int>" }
	%"struct.__gnu_cxx::hash_map<const llvm::Value*,int,__gnu_cxx::hash<const llvm::Value*>,std::equal_to<const llvm::Value*>,std::allocator<int> >" = type { %"struct.__gnu_cxx::hashtable<std::pair<const llvm::Value* const, int>,const llvm::Value*,__gnu_cxx::hash<const llvm::Value*>,std::_Select1st<std::pair<const llvm::Value* const, int> >,std::equal_to<const llvm::Value*>,std::allocator<int> >" }
	%"struct.__gnu_cxx::hash_set<const llvm::Constant*,__gnu_cxx::hash<const llvm::Constant*>,std::equal_to<const llvm::Constant*>,std::allocator<const llvm::Constant*> >" = type { %"struct.__gnu_cxx::hashtable<const llvm::Constant*,const llvm::Constant*,__gnu_cxx::hash<const llvm::Constant*>,std::_Identity<const llvm::Constant*>,std::equal_to<const llvm::Constant*>,std::allocator<const llvm::Constant*> >" }
	%"struct.__gnu_cxx::hashtable<const llvm::Constant*,const llvm::Constant*,__gnu_cxx::hash<const llvm::Constant*>,std::_Identity<const llvm::Constant*>,std::equal_to<const llvm::Constant*>,std::allocator<const llvm::Constant*> >" = type { %struct.__false_type, %struct.__false_type, %struct.__false_type, %struct.__false_type, %"struct.std::vector<__gnu_cxx::_Hashtable_node<const llvm::Constant*>*,std::allocator<const llvm::Constant*> >", i32 }
	%"struct.__gnu_cxx::hashtable<std::pair<const llvm::Value* const, int>,const llvm::Value*,__gnu_cxx::hash<const llvm::Value*>,std::_Select1st<std::pair<const llvm::Value* const, int> >,std::equal_to<const llvm::Value*>,std::allocator<int> >" = type { %struct.__false_type, %struct.__false_type, %struct.__false_type, %struct.__false_type, %"struct.std::vector<__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >*,std::allocator<int> >", i32 }
	%"struct.llvm::AbstractTypeUser" = type { i32 (...)** }
	%"struct.llvm::Annotable" = type { i32 (...)**, %"struct.llvm::Annotation"* }
	%"struct.llvm::Annotation" = type { i32 (...)**, %"struct.llvm::AnnotationID", %"struct.llvm::Annotation"* }
	%"struct.llvm::AnnotationID" = type { i32 }
	%"struct.llvm::Argument" = type { %"struct.llvm::Value", %"struct.llvm::Function"*, %"struct.llvm::Argument"*, %"struct.llvm::Argument"* }
	%"struct.llvm::BasicBlock" = type { %"struct.llvm::Value", %"struct.llvm::iplist<llvm::Instruction,llvm::ilist_traits<llvm::Instruction> >", %"struct.llvm::BasicBlock"*, %"struct.llvm::BasicBlock"* }
	%"struct.llvm::Constant" = type opaque
	%"struct.llvm::DerivedType" = type { %"struct.llvm::Type", %"struct.llvm::AbstractTypeUser", %"struct.std::vector<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" }
	%"struct.llvm::Function" = type { %"struct.llvm::GlobalValue", %"struct.llvm::Annotable", %"struct.llvm::iplist<llvm::BasicBlock,llvm::ilist_traits<llvm::BasicBlock> >", %"struct.llvm::iplist<llvm::Argument,llvm::ilist_traits<llvm::Argument> >", %"struct.llvm::SymbolTable"*, %"struct.llvm::Function"*, %"struct.llvm::Function"* }
	%"struct.llvm::FunctionPass" = type { %"struct.llvm::Pass" }
	%"struct.llvm::FunctionType" = type { %"struct.llvm::DerivedType", i1 }
	%"struct.llvm::GlobalValue" = type { %"struct.llvm::User", i32, %"struct.llvm::Module"* }
	%"struct.llvm::Instruction" = type { %"struct.llvm::User", %"struct.llvm::Annotable", %"struct.llvm::BasicBlock"*, %"struct.llvm::Instruction"*, %"struct.llvm::Instruction"*, i32 }
	%"struct.llvm::IntrinsicLowering" = type opaque
	%"struct.llvm::MachineBasicBlock" = type { %"struct.llvm::ilist<llvm::MachineInstr>", %"struct.llvm::MachineBasicBlock"*, %"struct.llvm::MachineBasicBlock"*, %"struct.llvm::BasicBlock"* }
	%"struct.llvm::MachineConstantPool" = type opaque
	%"struct.llvm::MachineFrameInfo" = type opaque
	%"struct.llvm::MachineFunction" = type { %"struct.llvm::Annotation", %"struct.llvm::Function"*, %"struct.llvm::TargetMachine"*, %"struct.llvm::iplist<llvm::MachineBasicBlock,llvm::ilist_traits<llvm::MachineBasicBlock> >", %"struct.llvm::SSARegMap"*, %"struct.llvm::MachineFunctionInfo"*, %"struct.llvm::MachineFrameInfo"*, %"struct.llvm::MachineConstantPool"* }
	%"struct.llvm::MachineFunctionInfo" = type { %"struct.__gnu_cxx::hash_set<const llvm::Constant*,__gnu_cxx::hash<const llvm::Constant*>,std::equal_to<const llvm::Constant*>,std::allocator<const llvm::Constant*> >", %"struct.__gnu_cxx::hash_map<const llvm::Value*,int,__gnu_cxx::hash<const llvm::Value*>,std::equal_to<const llvm::Value*>,std::allocator<int> >", i32, i32, i32, i32, i32, i32, i32, i1, i1, i1, %"struct.llvm::MachineFunction"* }
	%"struct.llvm::MachineFunctionPass" = type { %"struct.llvm::FunctionPass" }
	%"struct.llvm::MachineInstr" = type { i16, i8, %"struct.std::vector<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >", %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::MachineInstrBuilder" = type { %"struct.llvm::MachineInstr"* }
	%"struct.llvm::MachineOperand" = type { %"union.llvm::MachineOperand::._65", i32, i32 }
	%"struct.llvm::Module" = type opaque
	%"struct.llvm::PATypeHandle" = type { %"struct.llvm::Type"*, %"struct.llvm::AbstractTypeUser"* }
	%"struct.llvm::PATypeHolder" = type { %"struct.llvm::Type"* }
	%"struct.llvm::Pass" = type { i32 (...)**, %"struct.llvm::AbstractTypeUser"*, %"struct.llvm::PassInfo"*, %"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" }
	%"struct.llvm::PassInfo" = type { i8*, i8*, %"struct.std::type_info"*, i8, %"struct.std::vector<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >", %"struct.llvm::Pass"* ()*, %"struct.llvm::Pass"* (%"struct.llvm::TargetMachine"*)* }
	%"struct.llvm::SSARegMap" = type opaque
	%"struct.llvm::SymbolTable" = type opaque
	%"struct.llvm::SymbolTableListTraits<llvm::Argument,llvm::Function,llvm::Function,llvm::ilist_traits<llvm::Argument> >" = type { %"struct.llvm::Function"*, %"struct.llvm::Function"* }
	%"struct.llvm::SymbolTableListTraits<llvm::Instruction,llvm::BasicBlock,llvm::Function,llvm::ilist_traits<llvm::Instruction> >" = type { %"struct.llvm::Function"*, %"struct.llvm::BasicBlock"* }
	%"struct.llvm::TargetData" = type { %"struct.llvm::FunctionPass", i1, i8, i8, i8, i8, i8, i8, i8, i8 }
	%"struct.llvm::TargetFrameInfo" = type { i32 (...)**, i32, i32, i32 }
	%"struct.llvm::TargetInstrDescriptor" = type { i8*, i32, i32, i32, i1, i32, i32, i32, i32, i32, i32*, i32* }
	%"struct.llvm::TargetInstrInfo" = type { i32 (...)**, %"struct.llvm::TargetInstrDescriptor"*, i32, i32 }
	%"struct.llvm::TargetMachine" = type { i32 (...)**, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >", %"struct.llvm::TargetData", %"struct.llvm::IntrinsicLowering"* }
	%"struct.llvm::TargetRegClassInfo" = type { i32 (...)**, i32, i32, i32 }
	%"struct.llvm::TargetRegInfo" = type { i32 (...)**, %"struct.std::vector<const llvm::TargetRegClassInfo*,std::allocator<const llvm::TargetRegClassInfo*> >", %"struct.llvm::TargetMachine"* }
	%"struct.llvm::Type" = type { %"struct.llvm::Value", i32, i32, i1, i32, %"struct.llvm::Type"*, %"struct.std::vector<llvm::PATypeHandle,std::allocator<llvm::PATypeHandle> >" }
	%"struct.llvm::Use" = type { %"struct.llvm::Value"*, %"struct.llvm::User"*, %"struct.llvm::Use"*, %"struct.llvm::Use"* }
	%"struct.llvm::User" = type { %"struct.llvm::Value", %"struct.std::vector<llvm::Use,std::allocator<llvm::Use> >" }
	%"struct.llvm::Value" = type { i32 (...)**, %"struct.llvm::iplist<llvm::Use,llvm::ilist_traits<llvm::Use> >", %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >", %"struct.llvm::PATypeHolder", i32 }
	%"struct.llvm::_GLOBAL__N_::InsertPrologEpilogCode" = type { %"struct.llvm::MachineFunctionPass" }
	%"struct.llvm::ilist<llvm::MachineInstr>" = type { %"struct.llvm::iplist<llvm::MachineInstr,llvm::ilist_traits<llvm::MachineInstr> >" }
	%"struct.llvm::ilist_iterator<const llvm::MachineBasicBlock>" = type { %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::ilist_traits<llvm::Argument>" = type { %"struct.llvm::SymbolTableListTraits<llvm::Argument,llvm::Function,llvm::Function,llvm::ilist_traits<llvm::Argument> >" }
	%"struct.llvm::ilist_traits<llvm::Instruction>" = type { %"struct.llvm::SymbolTableListTraits<llvm::Instruction,llvm::BasicBlock,llvm::Function,llvm::ilist_traits<llvm::Instruction> >" }
	%"struct.llvm::iplist<llvm::Argument,llvm::ilist_traits<llvm::Argument> >" = type { %"struct.llvm::ilist_traits<llvm::Argument>", %"struct.llvm::Argument"*, %"struct.llvm::Argument"* }
	%"struct.llvm::iplist<llvm::BasicBlock,llvm::ilist_traits<llvm::BasicBlock> >" = type { %"struct.llvm::ilist_traits<llvm::Argument>", %"struct.llvm::BasicBlock"*, %"struct.llvm::BasicBlock"* }
	%"struct.llvm::iplist<llvm::Instruction,llvm::ilist_traits<llvm::Instruction> >" = type { %"struct.llvm::ilist_traits<llvm::Instruction>", %"struct.llvm::Instruction"*, %"struct.llvm::Instruction"* }
	%"struct.llvm::iplist<llvm::MachineBasicBlock,llvm::ilist_traits<llvm::MachineBasicBlock> >" = type { %"struct.llvm::MachineBasicBlock"*, %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::iplist<llvm::MachineInstr,llvm::ilist_traits<llvm::MachineInstr> >" = type { %"struct.llvm::ilist_iterator<const llvm::MachineBasicBlock>", %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineInstr"* }
	%"struct.llvm::iplist<llvm::Use,llvm::ilist_traits<llvm::Use> >" = type { %"struct.llvm::Use"*, %"struct.llvm::Use"* }
	%"struct.std::_Vector_alloc_base<__gnu_cxx::_Hashtable_node<const llvm::Constant*>*,std::allocator<const llvm::Constant*>, true>" = type { %"struct.__gnu_cxx::_Hashtable_node<const llvm::Constant*>"**, %"struct.__gnu_cxx::_Hashtable_node<const llvm::Constant*>"**, %"struct.__gnu_cxx::_Hashtable_node<const llvm::Constant*>"** }
	%"struct.std::_Vector_alloc_base<__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >*,std::allocator<int>, true>" = type { %"struct.__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >"**, %"struct.__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >"**, %"struct.__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >"** }
	%"struct.std::_Vector_alloc_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*>, true>" = type { %"struct.llvm::PassInfo"**, %"struct.llvm::PassInfo"**, %"struct.llvm::PassInfo"** }
	%"struct.std::_Vector_alloc_base<const llvm::TargetRegClassInfo*,std::allocator<const llvm::TargetRegClassInfo*>, true>" = type { %"struct.llvm::TargetFrameInfo"**, %"struct.llvm::TargetFrameInfo"**, %"struct.llvm::TargetFrameInfo"** }
	%"struct.std::_Vector_alloc_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*>, true>" = type { %"struct.llvm::AbstractTypeUser"**, %"struct.llvm::AbstractTypeUser"**, %"struct.llvm::AbstractTypeUser"** }
	%"struct.std::_Vector_alloc_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*>, true>" = type { %"struct.llvm::MachineInstr"**, %"struct.llvm::MachineInstr"**, %"struct.llvm::MachineInstr"** }
	%"struct.std::_Vector_alloc_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand>, true>" = type { %"struct.llvm::MachineOperand"*, %"struct.llvm::MachineOperand"*, %"struct.llvm::MachineOperand"* }
	%"struct.std::_Vector_alloc_base<llvm::PATypeHandle,std::allocator<llvm::PATypeHandle>, true>" = type { %"struct.llvm::PATypeHandle"*, %"struct.llvm::PATypeHandle"*, %"struct.llvm::PATypeHandle"* }
	%"struct.std::_Vector_alloc_base<llvm::Use,std::allocator<llvm::Use>, true>" = type { %"struct.llvm::Use"*, %"struct.llvm::Use"*, %"struct.llvm::Use"* }
	%"struct.std::_Vector_alloc_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> >, true>" = type { %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"*, %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"*, %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"* }
	%"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<const llvm::Constant*>*,std::allocator<const llvm::Constant*> >" = type { %"struct.std::_Vector_alloc_base<__gnu_cxx::_Hashtable_node<const llvm::Constant*>*,std::allocator<const llvm::Constant*>, true>" }
	%"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >*,std::allocator<int> >" = type { %"struct.std::_Vector_alloc_base<__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >*,std::allocator<int>, true>" }
	%"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" = type { %"struct.std::_Vector_alloc_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*>, true>" }
	%"struct.std::_Vector_base<const llvm::TargetRegClassInfo*,std::allocator<const llvm::TargetRegClassInfo*> >" = type { %"struct.std::_Vector_alloc_base<const llvm::TargetRegClassInfo*,std::allocator<const llvm::TargetRegClassInfo*>, true>" }
	%"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" = type { %"struct.std::_Vector_alloc_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*>, true>" }
	%"struct.std::_Vector_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" = type { %"struct.std::_Vector_alloc_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*>, true>" }
	%"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" = type { %"struct.std::_Vector_alloc_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand>, true>" }
	%"struct.std::_Vector_base<llvm::PATypeHandle,std::allocator<llvm::PATypeHandle> >" = type { %"struct.std::_Vector_alloc_base<llvm::PATypeHandle,std::allocator<llvm::PATypeHandle>, true>" }
	%"struct.std::_Vector_base<llvm::Use,std::allocator<llvm::Use> >" = type { %"struct.std::_Vector_alloc_base<llvm::Use,std::allocator<llvm::Use>, true>" }
	%"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" = type { %"struct.std::_Vector_alloc_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> >, true>" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" = type { i8* }
	%"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>" = type { %"struct.llvm::PassInfo"*, %"struct.llvm::Pass"* }
	%"struct.std::pair<const llvm::Value* const,int>" = type { %"struct.llvm::Value"*, i32 }
	%"struct.std::type_info" = type { i32 (...)**, i8* }
	%"struct.std::vector<__gnu_cxx::_Hashtable_node<const llvm::Constant*>*,std::allocator<const llvm::Constant*> >" = type { %"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<const llvm::Constant*>*,std::allocator<const llvm::Constant*> >" }
	%"struct.std::vector<__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >*,std::allocator<int> >" = type { %"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<std::pair<const llvm::Value* const, int> >*,std::allocator<int> >" }
	%"struct.std::vector<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" = type { %"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" }
	%"struct.std::vector<const llvm::TargetRegClassInfo*,std::allocator<const llvm::TargetRegClassInfo*> >" = type { %"struct.std::_Vector_base<const llvm::TargetRegClassInfo*,std::allocator<const llvm::TargetRegClassInfo*> >" }
	%"struct.std::vector<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" = type { %"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" }
	%"struct.std::vector<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" = type { %"struct.std::_Vector_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" }
	%"struct.std::vector<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" = type { %"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" }
	%"struct.std::vector<llvm::PATypeHandle,std::allocator<llvm::PATypeHandle> >" = type { %"struct.std::_Vector_base<llvm::PATypeHandle,std::allocator<llvm::PATypeHandle> >" }
	%"struct.std::vector<llvm::Use,std::allocator<llvm::Use> >" = type { %"struct.std::_Vector_base<llvm::Use,std::allocator<llvm::Use> >" }
	%"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" }
	%"union.llvm::MachineOperand::._65" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* }

declare void @_Znwj()

declare void @_ZN4llvm12MachineInstrC1Esjbb()

declare void @_ZNSt6vectorIPN4llvm12MachineInstrESaIS2_EE9push_backERKS2_()

declare void @_ZNK4llvm8Function15getFunctionTypeEv()

declare void @_ZNK4llvm19MachineInstrBuilder7addMRegEiNS_14MachineOperand7UseTypeE()

declare void @_ZNK4llvm19MachineInstrBuilder7addSImmEi()

define void @_ZN4llvm11_GLOBAL__N_22InsertPrologEpilogCode20runOnMachineFunctionERNS_15MachineFunctionE(%"struct.llvm::MachineFunction"* %F) {
entry:
	%tmp.8.i = invoke %"struct.llvm::TargetFrameInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.0.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetFrameInfo"*> [#uses=0]

invoke_catch.0.i:		; preds = %invoke_cont.49.i, %invoke_cont.48.i, %invoke_cont.47.i, %invoke_cont.i53.i, %no_exit.i, %invoke_cont.44.i, %invoke_cont.43.i, %invoke_cont.42.i, %invoke_cont.41.i, %invoke_cont.40.i, %invoke_cont.39.i, %invoke_cont.38.i, %invoke_cont.37.i, %then.2.i, %invoke_cont.35.i, %invoke_cont.34.i, %then.1.i, %endif.0.i, %invoke_cont.9.i, %invoke_cont.8.i, %invoke_cont.7.i, %invoke_cont.i.i, %then.0.i, %invoke_cont.4.i, %invoke_cont.3.i, %invoke_cont.2.i, %invoke_cont.1.i, %endif.0.i.i, %tmp.7.i.noexc.i, %invoke_cont.0.i, %entry
	ret void

invoke_cont.0.i:		; preds = %entry
	%tmp.7.i1.i = invoke %"struct.llvm::TargetFrameInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %tmp.7.i.noexc.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetFrameInfo"*> [#uses=2]

tmp.7.i.noexc.i:		; preds = %invoke_cont.0.i
	%tmp.17.i2.i = invoke i32 null( %"struct.llvm::TargetFrameInfo"* %tmp.7.i1.i )
			to label %endif.0.i.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

endif.0.i.i:		; preds = %tmp.7.i.noexc.i
	%tmp.38.i4.i = invoke i32 null( %"struct.llvm::TargetFrameInfo"* %tmp.7.i1.i )
			to label %tmp.38.i.noexc.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

tmp.38.i.noexc.i:		; preds = %endif.0.i.i
	br i1 false, label %invoke_cont.1.i, label %then.1.i.i

then.1.i.i:		; preds = %tmp.38.i.noexc.i
	ret void

invoke_cont.1.i:		; preds = %tmp.38.i.noexc.i
	%tmp.21.i = invoke %"struct.llvm::TargetRegInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.2.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetRegInfo"*> [#uses=1]

invoke_cont.2.i:		; preds = %invoke_cont.1.i
	%tmp.28.i = invoke i32 null( %"struct.llvm::TargetRegInfo"* %tmp.21.i )
			to label %invoke_cont.3.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

invoke_cont.3.i:		; preds = %invoke_cont.2.i
	%tmp.36.i = invoke %"struct.llvm::TargetInstrInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.4.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetInstrInfo"*> [#uses=1]

invoke_cont.4.i:		; preds = %invoke_cont.3.i
	%tmp.43.i = invoke i1 null( %"struct.llvm::TargetInstrInfo"* %tmp.36.i, i16 383, i64 0 )
			to label %invoke_cont.5.i unwind label %invoke_catch.0.i		; <i1> [#uses=1]

invoke_cont.5.i:		; preds = %invoke_cont.4.i
	br i1 %tmp.43.i, label %then.0.i, label %else.i

then.0.i:		; preds = %invoke_cont.5.i
	invoke void @_Znwj( )
			to label %tmp.0.i.noexc.i unwind label %invoke_catch.0.i

tmp.0.i.noexc.i:		; preds = %then.0.i
	invoke void @_ZN4llvm12MachineInstrC1Esjbb( )
			to label %invoke_cont.i.i unwind label %cond_true.i.i

cond_true.i.i:		; preds = %tmp.0.i.noexc.i
	ret void

invoke_cont.i.i:		; preds = %tmp.0.i.noexc.i
	invoke void @_ZNK4llvm19MachineInstrBuilder7addMRegEiNS_14MachineOperand7UseTypeE( )
			to label %invoke_cont.7.i unwind label %invoke_catch.0.i

invoke_cont.7.i:		; preds = %invoke_cont.i.i
	invoke void @_ZNK4llvm19MachineInstrBuilder7addSImmEi( )
			to label %invoke_cont.8.i unwind label %invoke_catch.0.i

invoke_cont.8.i:		; preds = %invoke_cont.7.i
	invoke void @_ZNK4llvm19MachineInstrBuilder7addMRegEiNS_14MachineOperand7UseTypeE( )
			to label %invoke_cont.9.i unwind label %invoke_catch.0.i

invoke_cont.9.i:		; preds = %invoke_cont.8.i
	invoke void @_ZNSt6vectorIPN4llvm12MachineInstrESaIS2_EE9push_backERKS2_( )
			to label %endif.0.i unwind label %invoke_catch.0.i

else.i:		; preds = %invoke_cont.5.i
	ret void

endif.0.i:		; preds = %invoke_cont.9.i
	invoke void @_ZNK4llvm8Function15getFunctionTypeEv( )
			to label %invoke_cont.33.i unwind label %invoke_catch.0.i

invoke_cont.33.i:		; preds = %endif.0.i
	br i1 false, label %then.1.i, label %endif.1.i

then.1.i:		; preds = %invoke_cont.33.i
	invoke void @_ZNK4llvm8Function15getFunctionTypeEv( )
			to label %invoke_cont.34.i unwind label %invoke_catch.0.i

invoke_cont.34.i:		; preds = %then.1.i
	%tmp.121.i = invoke %"struct.llvm::TargetRegInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.35.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetRegInfo"*> [#uses=1]

invoke_cont.35.i:		; preds = %invoke_cont.34.i
	%tmp.128.i = invoke i32 null( %"struct.llvm::TargetRegInfo"* %tmp.121.i )
			to label %invoke_cont.36.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

invoke_cont.36.i:		; preds = %invoke_cont.35.i
	br i1 false, label %then.2.i, label %endif.1.i

then.2.i:		; preds = %invoke_cont.36.i
	%tmp.140.i = invoke %"struct.llvm::TargetRegInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.37.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetRegInfo"*> [#uses=0]

invoke_cont.37.i:		; preds = %then.2.i
	%tmp.148.i = invoke %"struct.llvm::TargetRegInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.38.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetRegInfo"*> [#uses=1]

invoke_cont.38.i:		; preds = %invoke_cont.37.i
	%tmp.155.i = invoke i32 null( %"struct.llvm::TargetRegInfo"* %tmp.148.i, %"struct.llvm::Type"* null, i1 false )
			to label %invoke_cont.39.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

invoke_cont.39.i:		; preds = %invoke_cont.38.i
	%tmp.163.i = invoke %"struct.llvm::TargetFrameInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.40.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetFrameInfo"*> [#uses=1]

invoke_cont.40.i:		; preds = %invoke_cont.39.i
	%tmp.170.i = invoke i32 null( %"struct.llvm::TargetFrameInfo"* %tmp.163.i )
			to label %invoke_cont.41.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

invoke_cont.41.i:		; preds = %invoke_cont.40.i
	%tmp.177.i = invoke %"struct.llvm::TargetFrameInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.42.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetFrameInfo"*> [#uses=1]

invoke_cont.42.i:		; preds = %invoke_cont.41.i
	%tmp.184.i = invoke i32 null( %"struct.llvm::TargetFrameInfo"* %tmp.177.i )
			to label %invoke_cont.43.i unwind label %invoke_catch.0.i		; <i32> [#uses=1]

invoke_cont.43.i:		; preds = %invoke_cont.42.i
	%tmp.191.i = invoke %"struct.llvm::TargetFrameInfo"* null( %"struct.llvm::TargetMachine"* null )
			to label %invoke_cont.44.i unwind label %invoke_catch.0.i		; <%"struct.llvm::TargetFrameInfo"*> [#uses=1]

invoke_cont.44.i:		; preds = %invoke_cont.43.i
	%tmp.198.i = invoke i32 null( %"struct.llvm::TargetFrameInfo"* %tmp.191.i, %"struct.llvm::MachineFunction"* %F, i1* null )
			to label %invoke_cont.45.i unwind label %invoke_catch.0.i		; <i32> [#uses=0]

invoke_cont.45.i:		; preds = %invoke_cont.44.i
	br i1 false, label %no_exit.i, label %endif.1.i

no_exit.i:		; preds = %invoke_cont.50.i, %invoke_cont.45.i
	%nextArgOffset.0.i.1 = phi i32 [ %tmp.221.i, %invoke_cont.50.i ], [ 0, %invoke_cont.45.i ]		; <i32> [#uses=1]
	invoke void @_Znwj( )
			to label %tmp.0.i.noexc55.i unwind label %invoke_catch.0.i

tmp.0.i.noexc55.i:		; preds = %no_exit.i
	invoke void @_ZN4llvm12MachineInstrC1Esjbb( )
			to label %invoke_cont.i53.i unwind label %cond_true.i52.i

cond_true.i52.i:		; preds = %tmp.0.i.noexc55.i
	ret void

invoke_cont.i53.i:		; preds = %tmp.0.i.noexc55.i
	invoke void @_ZNK4llvm19MachineInstrBuilder7addMRegEiNS_14MachineOperand7UseTypeE( )
			to label %invoke_cont.47.i unwind label %invoke_catch.0.i

invoke_cont.47.i:		; preds = %invoke_cont.i53.i
	invoke void @_ZNK4llvm19MachineInstrBuilder7addMRegEiNS_14MachineOperand7UseTypeE( )
			to label %invoke_cont.48.i unwind label %invoke_catch.0.i

invoke_cont.48.i:		; preds = %invoke_cont.47.i
	invoke void @_ZNK4llvm19MachineInstrBuilder7addSImmEi( )
			to label %invoke_cont.49.i unwind label %invoke_catch.0.i

invoke_cont.49.i:		; preds = %invoke_cont.48.i
	invoke void @_ZNSt6vectorIPN4llvm12MachineInstrESaIS2_EE9push_backERKS2_( )
			to label %invoke_cont.50.i unwind label %invoke_catch.0.i

invoke_cont.50.i:		; preds = %invoke_cont.49.i
	%tmp.221.i = add i32 %nextArgOffset.0.i.1, %tmp.184.i		; <i32> [#uses=1]
	br i1 false, label %no_exit.i, label %endif.1.i

endif.1.i:		; preds = %invoke_cont.50.i, %invoke_cont.45.i, %invoke_cont.36.i, %invoke_cont.33.i
	ret void
}
