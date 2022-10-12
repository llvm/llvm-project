; RUN: opt -passes=dse -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.pair.162 = type { ptr, i32, [4 x i8] }
%struct.BasicBlock = type { %struct.Value, %struct.ilist_node.24, %struct.iplist.22, ptr }
%struct.Value = type { ptr, i8, i8, i16, ptr, ptr, ptr }
%struct.Type = type { ptr, i8, [3 x i8], i32, ptr }
%struct.LLVMContext = type { ptr }
%struct.LLVMContextImpl = type opaque
%struct.Use = type { ptr, ptr, %struct.PointerIntPair }
%struct.PointerIntPair = type { i64 }
%struct.StringMapEntry = type opaque
%struct.ilist_node.24 = type { %struct.ilist_half_node.23, ptr }
%struct.ilist_half_node.23 = type { ptr }
%struct.iplist.22 = type { %struct.ilist_traits.21, ptr }
%struct.ilist_traits.21 = type { %struct.ilist_half_node.25 }
%struct.ilist_half_node.25 = type { ptr }
%struct.Instruction = type { [52 x i8], %struct.ilist_node.26, ptr, %struct.DebugLoc }
%struct.ilist_node.26 = type { %struct.ilist_half_node.25, ptr }
%struct.DebugLoc = type { i32, i32 }
%struct.Function = type { %struct.GlobalValue, %struct.ilist_node.14, %struct.iplist.4, %struct.iplist, ptr, %struct.AttrListPtr }
%struct.GlobalValue = type <{ [52 x i8], [4 x i8], ptr, i8, i16, [5 x i8], %struct.basic_string }>
%struct.Module = type { ptr, %struct.iplist.20, %struct.iplist.16, %struct.iplist.12, %struct.vector.2, %struct.ilist, %struct.basic_string, ptr, %struct.OwningPtr, %struct.basic_string, %struct.basic_string, %struct.basic_string, ptr }
%struct.iplist.20 = type { %struct.ilist_traits.19, ptr }
%struct.ilist_traits.19 = type { %struct.ilist_node.18 }
%struct.ilist_node.18 = type { %struct.ilist_half_node.17, ptr }
%struct.ilist_half_node.17 = type { ptr }
%struct.GlobalVariable = type { %struct.GlobalValue, %struct.ilist_node.18, i8, [7 x i8] }
%struct.iplist.16 = type { %struct.ilist_traits.15, ptr }
%struct.ilist_traits.15 = type { %struct.ilist_node.14 }
%struct.ilist_node.14 = type { %struct.ilist_half_node.13, ptr }
%struct.ilist_half_node.13 = type { ptr }
%struct.iplist.12 = type { %struct.ilist_traits.11, ptr }
%struct.ilist_traits.11 = type { %struct.ilist_node.10 }
%struct.ilist_node.10 = type { %struct.ilist_half_node.9, ptr }
%struct.ilist_half_node.9 = type { ptr }
%struct.GlobalAlias = type { %struct.GlobalValue, %struct.ilist_node.10 }
%struct.vector.2 = type { %struct._Vector_base.1 }
%struct._Vector_base.1 = type { %struct._Vector_impl.0 }
%struct._Vector_impl.0 = type { ptr, ptr, ptr }
%struct.basic_string = type { %struct._Alloc_hider }
%struct._Alloc_hider = type { ptr }
%struct.ilist = type { %struct.iplist.8 }
%struct.iplist.8 = type { %struct.ilist_traits.7, ptr }
%struct.ilist_traits.7 = type { %struct.ilist_node.6 }
%struct.ilist_node.6 = type { %struct.ilist_half_node.5, ptr }
%struct.ilist_half_node.5 = type { ptr }
%struct.NamedMDNode = type { %struct.ilist_node.6, %struct.basic_string, ptr, ptr }
%struct.ValueSymbolTable = type opaque
%struct.OwningPtr = type { ptr }
%struct.GVMaterializer = type opaque
%struct.iplist.4 = type { %struct.ilist_traits.3, ptr }
%struct.ilist_traits.3 = type { %struct.ilist_half_node.23 }
%struct.iplist = type { %struct.ilist_traits, ptr }
%struct.ilist_traits = type { %struct.ilist_half_node }
%struct.ilist_half_node = type { ptr }
%struct.Argument = type { %struct.Value, %struct.ilist_node, ptr }
%struct.ilist_node = type { %struct.ilist_half_node, ptr }
%struct.AttrListPtr = type { ptr }
%struct.AttributeListImpl = type opaque

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

; CHECK: _ZSt9iter_swapIPSt4pairIPN4llvm10BasicBlockEjES5_EvT_T0_
; CHECK: store
; CHECK: ret void
define void @_ZSt9iter_swapIPSt4pairIPN4llvm10BasicBlockEjES5_EvT_T0_(ptr %__a, ptr %__b) nounwind uwtable inlinehint {
entry:
  %memtmp = alloca %struct.pair.162, align 8
  %0 = load ptr, ptr %__a, align 8
  store ptr %0, ptr %memtmp, align 8
  %1 = getelementptr inbounds %struct.pair.162, ptr %memtmp, i64 0, i32 1
  %2 = getelementptr inbounds %struct.pair.162, ptr %__a, i64 0, i32 1
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %1, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %__a, ptr %__b, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %__b, ptr %memtmp, i64 12, i1 false)
  ret void
}
