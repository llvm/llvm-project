; RUN: opt < %s -passes=globalopt -disable-output
; PR1491

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%"struct.__gnu_cxx::new_allocator<std::_Rb_tree_node<std::pair<const int, int> > >" = type <{ i8 }>
	%"struct.std::_Rb_tree<int,std::pair<const int, int>,std::_Select1st<std::pair<const int, int> >,std::less<int>,std::allocator<std::pair<const int, int> > >" = type { %"struct.std::_Rb_tree<int,std::pair<const int, int>,std::_Select1st<std::pair<const int, int> >,std::less<int>,std::allocator<std::pair<const int, int> > >::_Rb_tree_impl<std::less<int>,false>" }
	%"struct.std::_Rb_tree<int,std::pair<const int, int>,std::_Select1st<std::pair<const int, int> >,std::less<int>,std::allocator<std::pair<const int, int> > >::_Rb_tree_impl<std::less<int>,false>" = type { %"struct.__gnu_cxx::new_allocator<std::_Rb_tree_node<std::pair<const int, int> > >", %"struct.std::_Rb_tree_node_base", i32 }
	%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
	%"struct.std::map<int,int,std::less<int>,std::allocator<std::pair<const int, int> > >" = type { %"struct.std::_Rb_tree<int,std::pair<const int, int>,std::_Select1st<std::pair<const int, int> >,std::less<int>,std::allocator<std::pair<const int, int> > >" }
@someMap = global %"struct.std::map<int,int,std::less<int>,std::allocator<std::pair<const int, int> > >" zeroinitializer		; <%"struct.std::map<int,int,std::less<int>,std::allocator<std::pair<const int, int> > >"*> [#uses=1]
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_someMap, i8* null } ]		; <[1 x { i32, void ()*, i8* }]*> [#uses=0]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__D_someMap, i8* null } ]		; <[1 x { i32, void ()*, i8* }]*> [#uses=0]

define void @_GLOBAL__I_someMap() {
entry:
	call void @_Z41__static_initialization_and_destruction_0ii( i32 1, i32 65535 )
	ret void
}

declare void @_GLOBAL__D_someMap()

define void @_Z41__static_initialization_and_destruction_0ii(i32 %__initialize_p, i32 %__priority) {
entry:
	%tmp1 = icmp eq i32 %__priority, 65535		; <i1> [#uses=1]
	%tmp4 = icmp eq i32 %__initialize_p, 1		; <i1> [#uses=1]
	%tmp7 = and i1 %tmp1, %tmp4		; <i1> [#uses=1]
	br i1 %tmp7, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	store i8 0, i8* getelementptr (%"struct.std::map<int,int,std::less<int>,std::allocator<std::pair<const int, int> > >", %"struct.std::map<int,int,std::less<int>,std::allocator<std::pair<const int, int> > >"* @someMap, i32 0, i32 0, i32 0, i32 0, i32 0)
	ret void

cond_next:		; preds = %entry
	ret void
}
