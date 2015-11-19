; RUN: opt -dse -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.pair.162 = type { %struct.BasicBlock*, i32, [4 x i8] }
%struct.BasicBlock = type { %struct.Value, %struct.ilist_node.24, %struct.iplist.22, %struct.Function* }
%struct.Value = type { i32 (...)**, i8, i8, i16, %struct.Type*, %struct.Use*, %struct.StringMapEntry* }
%struct.Type = type { %struct.LLVMContext*, i8, [3 x i8], i32, {}* }
%struct.LLVMContext = type { %struct.LLVMContextImpl* }
%struct.LLVMContextImpl = type opaque
%struct.Use = type { %struct.Value*, %struct.Use*, %struct.PointerIntPair }
%struct.PointerIntPair = type { i64 }
%struct.StringMapEntry = type opaque
%struct.ilist_node.24 = type { %struct.ilist_half_node.23, %struct.BasicBlock* }
%struct.ilist_half_node.23 = type { %struct.BasicBlock* }
%struct.iplist.22 = type { %struct.ilist_traits.21, %struct.Instruction* }
%struct.ilist_traits.21 = type { %struct.ilist_half_node.25 }
%struct.ilist_half_node.25 = type { %struct.Instruction* }
%struct.Instruction = type { [52 x i8], %struct.ilist_node.26, %struct.BasicBlock*, %struct.DebugLoc }
%struct.ilist_node.26 = type { %struct.ilist_half_node.25, %struct.Instruction* }
%struct.DebugLoc = type { i32, i32 }
%struct.Function = type { %struct.GlobalValue, %struct.ilist_node.14, %struct.iplist.4, %struct.iplist, %struct.ValueSymbolTable*, %struct.AttrListPtr }
%struct.GlobalValue = type <{ [52 x i8], [4 x i8], %struct.Module*, i8, i16, [5 x i8], %struct.basic_string }>
%struct.Module = type { %struct.LLVMContext*, %struct.iplist.20, %struct.iplist.16, %struct.iplist.12, %struct.vector.2, %struct.ilist, %struct.basic_string, %struct.ValueSymbolTable*, %struct.OwningPtr, %struct.basic_string, %struct.basic_string, %struct.basic_string, i8* }
%struct.iplist.20 = type { %struct.ilist_traits.19, %struct.GlobalVariable* }
%struct.ilist_traits.19 = type { %struct.ilist_node.18 }
%struct.ilist_node.18 = type { %struct.ilist_half_node.17, %struct.GlobalVariable* }
%struct.ilist_half_node.17 = type { %struct.GlobalVariable* }
%struct.GlobalVariable = type { %struct.GlobalValue, %struct.ilist_node.18, i8, [7 x i8] }
%struct.iplist.16 = type { %struct.ilist_traits.15, %struct.Function* }
%struct.ilist_traits.15 = type { %struct.ilist_node.14 }
%struct.ilist_node.14 = type { %struct.ilist_half_node.13, %struct.Function* }
%struct.ilist_half_node.13 = type { %struct.Function* }
%struct.iplist.12 = type { %struct.ilist_traits.11, %struct.GlobalAlias* }
%struct.ilist_traits.11 = type { %struct.ilist_node.10 }
%struct.ilist_node.10 = type { %struct.ilist_half_node.9, %struct.GlobalAlias* }
%struct.ilist_half_node.9 = type { %struct.GlobalAlias* }
%struct.GlobalAlias = type { %struct.GlobalValue, %struct.ilist_node.10 }
%struct.vector.2 = type { %struct._Vector_base.1 }
%struct._Vector_base.1 = type { %struct._Vector_impl.0 }
%struct._Vector_impl.0 = type { %struct.basic_string*, %struct.basic_string*, %struct.basic_string* }
%struct.basic_string = type { %struct._Alloc_hider }
%struct._Alloc_hider = type { i8* }
%struct.ilist = type { %struct.iplist.8 }
%struct.iplist.8 = type { %struct.ilist_traits.7, %struct.NamedMDNode* }
%struct.ilist_traits.7 = type { %struct.ilist_node.6 }
%struct.ilist_node.6 = type { %struct.ilist_half_node.5, %struct.NamedMDNode* }
%struct.ilist_half_node.5 = type { %struct.NamedMDNode* }
%struct.NamedMDNode = type { %struct.ilist_node.6, %struct.basic_string, %struct.Module*, i8* }
%struct.ValueSymbolTable = type opaque
%struct.OwningPtr = type { %struct.GVMaterializer* }
%struct.GVMaterializer = type opaque
%struct.iplist.4 = type { %struct.ilist_traits.3, %struct.BasicBlock* }
%struct.ilist_traits.3 = type { %struct.ilist_half_node.23 }
%struct.iplist = type { %struct.ilist_traits, %struct.Argument* }
%struct.ilist_traits = type { %struct.ilist_half_node }
%struct.ilist_half_node = type { %struct.Argument* }
%struct.Argument = type { %struct.Value, %struct.ilist_node, %struct.Function* }
%struct.ilist_node = type { %struct.ilist_half_node, %struct.Argument* }
%struct.AttrListPtr = type { %struct.AttributeListImpl* }
%struct.AttributeListImpl = type opaque

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

; CHECK: _ZSt9iter_swapIPSt4pairIPN4llvm10BasicBlockEjES5_EvT_T0_
; CHECK: store
; CHECK: ret void
define void @_ZSt9iter_swapIPSt4pairIPN4llvm10BasicBlockEjES5_EvT_T0_(%struct.pair.162* %__a, %struct.pair.162* %__b) nounwind uwtable inlinehint {
entry:
  %memtmp = alloca %struct.pair.162, align 8
  %0 = getelementptr inbounds %struct.pair.162, %struct.pair.162* %memtmp, i64 0, i32 0
  %1 = getelementptr inbounds %struct.pair.162, %struct.pair.162* %__a, i64 0, i32 0
  %2 = load %struct.BasicBlock*, %struct.BasicBlock** %1, align 8
  store %struct.BasicBlock* %2, %struct.BasicBlock** %0, align 8
  %3 = getelementptr inbounds %struct.pair.162, %struct.pair.162* %memtmp, i64 0, i32 1
  %4 = getelementptr inbounds %struct.pair.162, %struct.pair.162* %__a, i64 0, i32 1
  %5 = load i32, i32* %4, align 4
  store i32 %5, i32* %3, align 8
  %6 = bitcast %struct.pair.162* %__a to i8*
  %7 = bitcast %struct.pair.162* %__b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %6, i8* %7, i64 12, i1 false)
  %8 = bitcast %struct.pair.162* %memtmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %7, i8* %8, i64 12, i1 false)
  ret void
}
