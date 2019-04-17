; RUN: opt < %s -simplifycfg -S | not grep "br label"

	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.global_variable.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%struct.Group = type { %struct.Scene, %struct.Sphere, %"struct.std::list<Scene*,std::allocator<Scene*> >" }
	%struct.Ray = type { %struct.Vec, %struct.Vec }
	%struct.Scene = type { i32 (...)** }
	%struct.Sphere = type { %struct.Scene, %struct.Vec, double }
	%struct.Vec = type { double, double, double }
	%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
	%struct.__false_type = type <{ i8 }>
	%"struct.__gnu_cxx::new_allocator<Scene*>" = type <{ i8 }>
	%"struct.__gnu_cxx::new_allocator<std::_List_node<Scene*> >" = type <{ i8 }>
	%struct.__si_class_type_info_pseudo = type { %struct.__type_info_pseudo, %"struct.std::type_info"* }
	%struct.__type_info_pseudo = type { i8*, i8* }
	%"struct.std::Hit" = type { double, %struct.Vec }
	%"struct.std::_List_base<Scene*,std::allocator<Scene*> >" = type { %"struct.std::_List_base<Scene*,std::allocator<Scene*> >::_List_impl" }
	%"struct.std::_List_base<Scene*,std::allocator<Scene*> >::_List_impl" = type { %"struct.std::_List_node_base" }
	%"struct.std::_List_const_iterator<Scene*>" = type { %"struct.std::_List_node_base"* }
	%"struct.std::_List_iterator<Scene*>" = type { %"struct.std::_List_node_base"* }
	%"struct.std::_List_node<Scene*>" = type { %"struct.std::_List_node_base", %struct.Scene* }
	%"struct.std::_List_node_base" = type { %"struct.std::_List_node_base"*, %"struct.std::_List_node_base"* }
	%"struct.std::allocator<Scene*>" = type <{ i8 }>
	%"struct.std::allocator<std::_List_node<Scene*> >" = type <{ i8 }>
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"* }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", i32*, i8, i32*, i32*, i32*, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
	%"struct.std::ios_base::Init" = type <{ i8 }>
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::list<Scene*,std::allocator<Scene*> >" = type { %"struct.std::_List_base<Scene*,std::allocator<Scene*> >" }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
	%"struct.std::num_put<char,std::ostreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
	%"struct.std::numeric_limits<double>" = type <{ i8 }>
	%"struct.std::type_info" = type { i32 (...)**, i8* }
@llvm.dbg.subprogram947 = external constant %llvm.dbg.subprogram.type		; <%llvm.dbg.subprogram.type*> [#uses=1]

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind

declare void @_ZN9__gnu_cxx13new_allocatorIP5SceneED2Ev(%struct.__false_type*) nounwind

define void @_ZNSaIP5SceneED1Ev(%struct.__false_type* %this) nounwind {
entry:
	%this_addr = alloca %struct.__false_type*		; <%struct.__false_type**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram947 to { }*))
	store %struct.__false_type* %this, %struct.__false_type** %this_addr
	%0 = load %struct.__false_type*, %struct.__false_type** %this_addr, align 4		; <%struct.__false_type*> [#uses=1]
	call void @_ZN9__gnu_cxx13new_allocatorIP5SceneED2Ev(%struct.__false_type* %0) nounwind
	br label %bb

bb:		; preds = %entry
	br label %return

return:		; preds = %bb
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram947 to { }*))
	ret void
}
