; RUN: opt < %s -globalopt -disable-output
; PR820
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct..0FileDescriptor = type { i32 }
	%"struct.FlagDescription<int32>" = type { i8*, i32*, i1, i1, i32, i8* }
	%"struct.FlagRegisterer<bool>" = type { i8 }
	%struct.MutexLock = type { %struct..0FileDescriptor* }
	%"struct.std::DisabledRangeMap" = type { %"struct.std::_Rb_tree<const char*,std::pair<const char* const, FlagDescription<bool> >,std::_Select1st<std::pair<const char* const, FlagDescription<bool> > >,StringCmp,std::allocator<std::pair<const char* const, FlagDescription<bool> > > >" }
	%"struct.std::_Rb_tree<const char*,std::pair<const char* const, FlagDescription<bool> >,std::_Select1st<std::pair<const char* const, FlagDescription<bool> > >,StringCmp,std::allocator<std::pair<const char* const, FlagDescription<bool> > > >" = type { %"struct.std::_Rb_tree<const char*,std::pair<const char* const, FlagDescription<bool> >,std::_Select1st<std::pair<const char* const, FlagDescription<bool> > >,StringCmp,std::allocator<std::pair<const char* const, FlagDescription<bool> > > >::_Rb_tree_impl<StringCmp,false>" }
	%"struct.std::_Rb_tree<const char*,std::pair<const char* const, FlagDescription<bool> >,std::_Select1st<std::pair<const char* const, FlagDescription<bool> > >,StringCmp,std::allocator<std::pair<const char* const, FlagDescription<bool> > > >::_Rb_tree_impl<StringCmp,false>" = type { %"struct.FlagRegisterer<bool>", %"struct.std::_Rb_tree_node_base", i32 }
	%"struct.std::_Rb_tree_const_iterator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" = type { %"struct.std::_Rb_tree_node_base"* }
	%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
	%"struct.std::_Vector_base<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" }
	%"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" = type { i32*, i32*, i32* }
	%"struct.std::vector<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >" }
@registry_lock = external global %struct..0FileDescriptor		; <%struct..0FileDescriptor*> [#uses=0]
@_ZN61FLAG__foo_int32_44FLAGS_E = external global %"struct.FlagRegisterer<bool>"		; <%"struct.FlagRegisterer<bool>"*> [#uses=0]
@llvm.global_ctors = appending global [20 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN62FLAG__foo_string_10FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN60FLAG__foo_bool_19FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZNK5Bzh4Enum13is_contiguousEv, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN62FLAG__foo_string_17FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN61FLAG__foo_int32_21FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN7ScannerC2Ev, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__Z11StripStringPSsPKcc, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZNK9__gnu_cxx4hashI11StringPieceEclERKS1_, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN8Hasher325ResetEj, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__Z25ACLRv, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN61FLAG__foo_int64_25FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN61FLAG__foo_int32_7FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN62FLAG__foo_string_18FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN62FLAG__foo_string_17FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN61FLAG__foo_int32_25FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_eventbuf, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN61FLAG__foo_int32_26FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN62FLAG__foo_string_16FLAGS_E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__ZN17InitializerC2EPKcS1_PFvvE, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__checker_bcad_variable, i8* null } ]		; <[20 x { i32, void ()*, i8* }]*> [#uses=0]

declare void @_GLOBAL__I__ZN62FLAG__foo_string_10FLAGS_E()

declare void @_GLOBAL__I__ZN60FLAG__foo_bool_19FLAGS_E()

declare void @_GLOBAL__I__ZNK5Bzh4Enum13is_contiguousEv()

declare void @_GLOBAL__I__ZN62FLAG__foo_string_17FLAGS_E()

declare void @_GLOBAL__I__ZN61FLAG__foo_int32_21FLAGS_E()

define void @_ZN14FlagRegistererIiEC1EPKcRK15FlagDescriptionIiE() {
entry:
	call void @_Z12RegisterFlagIiEvPKcRK15FlagDescriptionIT_E( )
	ret void
}

define void @_Z12RegisterFlagIiEvPKcRK15FlagDescriptionIT_E() {
entry:
	call void @_ZN9MutexLockC1EP5Mutex( )
	ret void
}

declare void @_GLOBAL__I__ZN7ScannerC2Ev()

declare void @_GLOBAL__I__Z11StripStringPSsPKcc()

define void @_ZNSt6vectorIiSaIiEEC1ERKS0_() {
entry:
	unreachable
}

declare void @_GLOBAL__I__ZNK9__gnu_cxx4hashI11StringPieceEclERKS1_()

declare void @_GLOBAL__I__ZN8Hasher325ResetEj()

declare void @_GLOBAL__I__Z25ACLRv()

define void @_ZN9MutexLockC1EP5Mutex() {
entry:
	call void @_ZN5Mutex4LockEv( )
	ret void
}

define void @_ZN5Mutex4LockEv() {
entry:
	call void @_Z22Acquire_CASPViii( )
	ret void
}

define void @_ZNSt3mapIPKc15FlagDescriptionIiE9StringCmpSaISt4pairIKS1_S3_EEE3endEv(%"struct.std::_Rb_tree_const_iterator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >"* sret  %agg.result) {
entry:
	unreachable
}

declare void @_GLOBAL__I__ZN61FLAG__foo_int64_25FLAGS_E()

define void @_Z14CASPViii() {
entry:
	%tmp3 = call i32 asm sideeffect "lock; cmpxchg $1,$2", "={ax},q,m,0,~{dirflag},~{fpsr},~{flags},~{memory}"( i32 0, i32* null, i32 0 )		; <i32> [#uses=0]
	unreachable
}

declare void @_GLOBAL__I__ZN61FLAG__foo_int32_7FLAGS_E()

declare void @_GLOBAL__I__ZN62FLAG__foo_string_18FLAGS_E()

define void @_Z22Acquire_CASPViii() {
entry:
	call void @_Z14CASPViii( )
	unreachable
}

declare void @_GLOBAL__I__ZN61FLAG__foo_int32_25FLAGS_E()

declare void @_GLOBAL__I_eventbuf()

define void @_GLOBAL__I__ZN61FLAG__foo_int32_26FLAGS_E() {
entry:
	call void @_Z41__static_initialization_and_destruction_0ii1662( i32 1, i32 65535 )
	ret void
}

define void @_Z41__static_initialization_and_destruction_0ii1662(i32 %__initialize_p, i32 %__priority) {
entry:
	%__initialize_p_addr = alloca i32		; <i32*> [#uses=2]
	%__priority_addr = alloca i32		; <i32*> [#uses=2]
	store i32 %__initialize_p, i32* %__initialize_p_addr
	store i32 %__priority, i32* %__priority_addr
	%tmp = load i32, i32* %__priority_addr		; <i32> [#uses=1]
	%tmp.upgrd.1 = icmp eq i32 %tmp, 65535		; <i1> [#uses=1]
	br i1 %tmp.upgrd.1, label %cond_true, label %cond_next14

cond_true:		; preds = %entry
	%tmp8 = load i32, i32* %__initialize_p_addr		; <i32> [#uses=1]
	%tmp9 = icmp eq i32 %tmp8, 1		; <i1> [#uses=1]
	br i1 %tmp9, label %cond_true10, label %cond_next14

cond_true10:		; preds = %cond_true
	call void @_ZN14FlagRegistererIiEC1EPKcRK15FlagDescriptionIiE( )
	ret void

cond_next14:		; preds = %cond_true, %entry
	ret void
}

declare void @_GLOBAL__I__ZN62FLAG__foo_string_16FLAGS_E()

define void @_ZN9__gnu_cxx13new_allocatorIPNS_15_Hashtable_nodeIjEEEC2Ev() {
entry:
	unreachable
}

declare void @_GLOBAL__I__ZN17InitializerC2EPKcS1_PFvvE()

declare void @_GLOBAL__I__checker_bcad_variable()
