//===-- NVVMOps.td - NVVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the NVVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_OPS
#define NVVMIR_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLVM_i8Ptr_global : LLVM_IntPtrBase<8, 1>;
def LLVM_i8Ptr_shared : LLVM_IntPtrBase<8, 3>;

//===----------------------------------------------------------------------===//
// NVVM dialect definitions
//===----------------------------------------------------------------------===//

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "::mlir::NVVM";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "nvvm.kernel"; }
  }];

  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// NVVM op definitions
//===----------------------------------------------------------------------===//

class NVVM_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// NVVM intrinsic operations
//===----------------------------------------------------------------------===//

class NVVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<Trait> traits,
                  int numResults>
  : LLVM_IntrOpBase<NVVM_Dialect, mnem, "nvvm_" # !subst(".", "_", mnem),
                    overloadedResults, overloadedOperands, traits, numResults>;


//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_SpecialRegisterOp<string mnemonic,
    list<Trait> traits = []> :
  NVVM_IntrOp<mnemonic, [], [], !listconcat(traits, [NoSideEffect]), 1>,
  Arguments<(ins)> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Lane index and range
def NVVM_LaneIdOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.laneid">;
def NVVM_WarpSizeOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.warpsize">;

//===----------------------------------------------------------------------===//
// Thread index and range
def NVVM_ThreadIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_BlockDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_BlockDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_BlockDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.z">;

//===----------------------------------------------------------------------===//
// Block index and range
def NVVM_BlockIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_GridDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_GridDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_GridDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.z">;

//===----------------------------------------------------------------------===//
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  }];
  let assemblyFormat = "attr-dict";
}

def ShflKindBfly : I32EnumAttrCase<"bfly", 0>;
def ShflKindUp   : I32EnumAttrCase<"up", 1>;
def ShflKindDown : I32EnumAttrCase<"down", 2>;
def ShflKindIdx  : I32EnumAttrCase<"idx", 3>;

/// Enum attribute of the different shuffle kinds.
def ShflKind : I32EnumAttr<"ShflKind", "NVVM shuffle kind",
  [ShflKindBfly, ShflKindUp, ShflKindDown, ShflKindIdx]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ShflKindAttr : EnumAttr<NVVM_Dialect, ShflKind, "shfl_kind">;

def NVVM_ShflOp :
  NVVM_Op<"shfl.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins I32:$dst,
                 LLVM_Type:$val,
                 I32:$offset,
                 I32:$mask_and_clamp,
                 ShflKindAttr:$kind,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  string llvmBuilder = [{
      auto intId = getShflIntrinsicId(
          $_resultType, $kind, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$dst, $val, $offset, $mask_and_clamp});
  }];
  let assemblyFormat = [{
    $kind $dst `,` $val `,` $offset `,` $mask_and_clamp  attr-dict
     `:` type($val) `->` type($res)
   }];
   let hasVerifier = 1;
}

def NVVM_VoteBallotOp :
  NVVM_Op<"vote.ballot.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$mask, LLVM_Type:$pred)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {$mask, $pred});
  }];
  let hasCustomAssemblyFormat = 1;
}


def NVVM_CpAsyncOp : NVVM_Op<"cp.async.shared.global">,
  Arguments<(ins LLVM_i8Ptr_shared:$dst,
                 LLVM_i8Ptr_global:$src,
                 I32Attr:$size)> {
  string llvmBuilder = [{
      llvm::Intrinsic::ID id;
      switch ($size) {
        case 4:
          id = llvm::Intrinsic::nvvm_cp_async_ca_shared_global_4;
          break;
        case 8:
          id = llvm::Intrinsic::nvvm_cp_async_ca_shared_global_8;
          break;
        case 16:
          id = llvm::Intrinsic::nvvm_cp_async_ca_shared_global_16;
          break;
        default:
          llvm_unreachable("unsupported async copy size");
      }
      createIntrinsicCall(builder, id, {$dst, $src});
  }];
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
  let hasVerifier = 1;
}

def NVVM_CpAsyncCommitGroupOp : NVVM_Op<"cp.async.commit.group"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_cp_async_commit_group);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_CpAsyncWaitGroupOp : NVVM_Op<"cp.async.wait.group">,
  Arguments<(ins I32Attr:$n)> {
  string llvmBuilder = [{
      createIntrinsicCall(
        builder,
        llvm::Intrinsic::nvvm_cp_async_wait_group,
        llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(moduleTranslation.getLLVMContext()),
          $n));
  }];
  let assemblyFormat = "$n attr-dict";
}

def NVVM_MmaOp :
  NVVM_Op<"mma.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  string llvmBuilder = [{
    $res = createIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f32, $args);
  }];
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
  let hasVerifier = 1;
}

/// Helpers to instantiate different version of wmma intrinsics.
/// This matches the hierarchy used in IntrinsicsNVVM.td to define all the
/// combinations of the intrinsics.
class GEOM<int M, int N, int K> {
  int m = M;
  int n = N;
  int k = K;
}

/// Class containing information about valid mma matrix types.
class WMMA_REGS<GEOM Geom, string Frag, string PtxEltType> {
  int m = Geom.m;
  int n = Geom.n;
  int k = Geom.k;
  string geom = "m"#Geom.m#"n"#Geom.n#"k"#Geom.k;
  string frag = Frag;
  string ptx_elt_type = PtxEltType;
  string gft = geom#":"#Frag#":"#ptx_elt_type;
}

//// Generate enum value of the mma.load/mma.store intrinsic.
class WMMA_NAME_LDST<string Op, WMMA_REGS Frag, string Layout, int WithStride> {
  string id =   "llvm::Intrinsic::nvvm_wmma"
                # "_" # Frag.geom
                # "_" # Op
                # "_" # Frag.frag
                # "_" # Frag.ptx_elt_type
                # "_" # Layout
                # !if(WithStride, "_stride", "");
}

/// Generate the signature part of the mma intrinsic name.
class MMA_SIGNATURE<WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  list<WMMA_REGS> id_frags = !cond(
     // FP16 ops are identified by accumulator & result type.
     !eq(A.ptx_elt_type, "f16") : [D, C],
     // other ops are identified by input types.
     !ne(A.ptx_elt_type, B.ptx_elt_type): [A, B],
     true: [A]
     );
   string ret = !foldl("", id_frags, a, b, !strconcat(a, "_", b.ptx_elt_type));
}

/// Generate enum value of the wmma.mma intrinsic.
class WMMA_NAME<string Op, string ALayout, string BLayout, WMMA_REGS A,
  WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  string signature = MMA_SIGNATURE<A, B, C, D>.ret;
  string id =   "llvm::Intrinsic::nvvm_wmma"
                # "_" # A.geom
                # "_" # Op
                # "_" # ALayout
                # "_" # BLayout
                # signature;
}

// Generates list of 4-tuples of WMMA_REGS representing a valid MMA op.
//   Geom: list of supported geometries.
//   TypeN: PTX type of the corresponding fragment's element.
//   TypeB and TypeD may be empty if it must match that of TypeA or TypeC.
class MMA_OPS<list<GEOM> Geom, list<string> TypeA, list<string> TypeB,
            list<string> TypeC, list<string> TypeD> {
  list<list<WMMA_REGS>> ret =
     !foldl([]<list<WMMA_REGS>>, Geom, t1, geom, !listconcat(t1,
     !foldl([]<list<WMMA_REGS>>, TypeA, t2, type_a, !listconcat(t2,
     !foldl([]<list<WMMA_REGS>>, !if(!size(TypeB), TypeB, [type_a]), t3, type_b, !listconcat(t3,
     !foldl([]<list<WMMA_REGS>>, TypeC, t4, type_c, !listconcat(t4,
     !foldl([]<list<WMMA_REGS>>, !if(!size(TypeD), TypeD, [type_c]), t5, type_d, !listconcat(t5,
            [[WMMA_REGS<geom, "a", type_a>,
              WMMA_REGS<geom, "b", type_b>,
              WMMA_REGS<geom, "c", type_c>,
              WMMA_REGS<geom, "d", type_d>]]))))))))));
   // Debugging aid for readable representation of the list above.
   list<list<string>> ops = !foreach(x, ret, [x[0].gft, x[1].gft, x[2].gft, x[3].gft]);
}

/// Creates a list of combinations of load/store operations supported.
class MMA_LDST_OPS<list<GEOM> Geom, list<string> Frags, list<string> Types> {
  list<WMMA_REGS> ret =
     !foldl([]<WMMA_REGS>, Geom, t1, geom, !listconcat(t1,
     !foldl([]<WMMA_REGS>, Frags, t2, frag, !listconcat(t2,
     !foldl([]<WMMA_REGS>, Types, t3, type, !listconcat(t3,
            [WMMA_REGS<geom, frag, type>]))))));
   // Debugging aid for readable representation of the list above.
   list<string> ops = !foreach(x, ret, x.gft);
}

// Creates list of valid combinations of fragments. This is a subset of what
// llvm supports and can be extended as needed.
class NVVM_MMA_OPS {
  list<list<WMMA_REGS>> tf32_wmma_ops = MMA_OPS<
            [GEOM<16, 16, 8>],
            ["tf32"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> fp_wmma_ops = MMA_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["f16"], [], ["f16", "f32"], []>.ret;
  list<list<WMMA_REGS>> all_wmma_ops = !listconcat(
            tf32_wmma_ops,
            fp_wmma_ops);

  list<WMMA_REGS> ldst_ab_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["a", "b"], ["f16"]>.ret;
  list<WMMA_REGS> ldst_cd_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["c", "d"], ["f16", "f32"]>.ret;
  list<WMMA_REGS> ldst_tf32_ab_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 8>],
            ["a", "b"], ["tf32"]>.ret;
  list<WMMA_REGS> ldst_tf32_cd_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 8>],
            ["c", "d"], ["f32"]>.ret;
  list<WMMA_REGS> all_ldst_ops = !listconcat(ldst_ab_ops, ldst_cd_ops,
                                             ldst_tf32_ab_ops,
                                             ldst_tf32_cd_ops);
  // Separate A/B/C fragments (loads) from D (stores).
  list<WMMA_REGS> all_ld_ops = !filter(op, all_ldst_ops, !ne(op.frag, "d"));
  list<WMMA_REGS> all_st_ops = !filter(op, all_ldst_ops, !eq(op.frag, "d"));
}

def NVVM_MMA_OPS : NVVM_MMA_OPS;

/// Helper to create the mapping between the configuration and the store
/// intrinsic enum value.
class MMA_ST_INTR<string op> {
  list<list<string>> cond0 = !foreach(frag, NVVM_MMA_OPS.all_st_ops,
                                !foreach(layout, ["row", "col"],
  "if (layout == \"" # layout #  "\" && m == " # frag.m # " &&"
  "    n == " #frag.n # " && k == " # frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return " #WMMA_NAME_LDST<op, frag, layout, 1>.id #";"));
  string id = !foldl("",
                !foldl([""], cond0, acc, el, !listconcat(acc, el)),
                acc1, el1, acc1 # "\n" # el1);
}

/// Helper to map a mxk shape to a supported mxnxk matrix type. This will return
/// the n value of the supported configuration.
class MMA_ST_INFER_N<list<WMMA_REGS> ldst> {
  list<string> cond = !foreach(frag, ldst,
  "if (m == " # frag.m # " && k == " #frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return "# frag.n #";");
  string id = !foldl("", cond, acc, el, acc # "\n" # el);
}

/// Helper to map a kxn shape to a supported mxnxk matrix type. This will return
/// the m value of the supported configuration.
class MMA_ST_INFER_M<list<WMMA_REGS> ldst> {
  list<string> cond = !foreach(frag, ldst,
  "if (n == " # frag.n # " && k == " #frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return "# frag.m #";");
  string id = !foldl("", cond, acc, el, acc # "\n" # el);
}

/// Helper to map a mxn shape to a supported mxnxk matrix type. This will return
/// the k value of the supported configuration.
class MMA_ST_INFER_K<list<WMMA_REGS> ldst> {
  list<string> cond = !foreach(frag, ldst,
  "if (m == " # frag.m # " && n == " #frag.n # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return "# frag.k #";");
  string id = !foldl("", cond, acc, el, acc # "\n" # el);
}

/// Helper to create the mapping between the configuration and the load
/// intrinsic enum value.
class MMA_LD_INTR<string op> {
  list<list<string>> cond0 = !foreach(frag, NVVM_MMA_OPS.all_ld_ops,
                                !foreach(layout, ["row", "col"],
  "if (layout == \"" # layout #  "\" && m == " # frag.m # " &&"
  "    n == " #frag.n # " && k == " # frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype && frag == \""#frag.frag#"\")"
  "  return "# WMMA_NAME_LDST<op, frag, layout, 1>.id #";"));
  string id = !foldl("",
                !foldl([""], cond0, acc, el, !listconcat(acc, el)),
                acc1, el1, acc1 # "\n" # el1);
}

/// Helper to create the mapping between the configuration and the wmma.mma
/// intrinsic enum value.
class MMA_MMA_INTR<string opName> {
  list<list<list<string>>> cond0 =
    !foreach(op, NVVM_MMA_OPS.all_wmma_ops,
      !foreach(layoutA, ["row", "col"],
        !foreach(layoutB, ["row", "col"],
  "if (layoutA == \"" # layoutA #  "\" && layoutB == \"" # layoutB #  "\" && "
  "    m == " # op[0].m # " && n == " #op[0].n # " && k == " # op[0].k #
  "    && \"" # op[0].ptx_elt_type # "\" == eltypeA && \""
   # op[3].ptx_elt_type # "\" == eltypeB)"
  "  return " #
       WMMA_NAME<opName, layoutA, layoutB, op[0], op[1], op[2], op[3]>.id # ";")));
  list<string> f = !foldl([""],
                     !foldl([[""]], cond0, acc, el, !listconcat(acc, el)),
                          acc1, el1, !listconcat(acc1, el1));
  string id = !foldl("", f, acc, el, acc # "\n" # el);
}

def MMALayoutRow : I32EnumAttrCase<"row", 0>;
def MMALayoutCol : I32EnumAttrCase<"col", 1>;

/// Enum attribute of the different matrix layout.
def MMALayout : I32EnumAttr<"MMALayout", "NVVM MMA layout",
  [MMALayoutRow, MMALayoutCol]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMALayoutAttr : EnumAttr<NVVM_Dialect, MMALayout, "mma_layout"> {
  let assemblyFormat = "`<` $value `>`";
}

def MMATypeF16  : I32EnumAttrCase<"f16", 0>;
def MMATypeF32  : I32EnumAttrCase<"f32", 1>;
def MMATypeTF32 : I32EnumAttrCase<"tf32", 2>;

/// Enum attribute of the different matrix types.
def MMATypes : I32EnumAttr<"MMATypes", "NVVM MMA types",
  [MMATypeF16, MMATypeF32, MMATypeTF32]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMATypesAttr : EnumAttr<NVVM_Dialect, MMATypes, "mma_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def MMAFragA : I32EnumAttrCase<"a", 0>;
def MMAFragB : I32EnumAttrCase<"b", 1>;
def MMAFragC : I32EnumAttrCase<"c", 2>;

/// Enum attribute of the different frag types.
def MMAFrag: I32EnumAttr<"MMAFrag", "NVVM MMA frag type",
  [MMAFragA, MMAFragB, MMAFragC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAFragAttr : EnumAttr<NVVM_Dialect, MMAFrag, "mma_frag"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_WMMALoadOp: NVVM_Op<"wmma.load">,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins LLVM_AnyPointer: $ptr, I32: $stride, I32Attr:$m,
             I32Attr:$n, I32Attr:$k, MMALayoutAttr:$layout,
             MMATypesAttr:$eltype, MMAFragAttr:$frag)> {

  let summary = "Warp synchronous matrix load";

  // Since LLVM intrinsic IDs are enum that cannot be dynamically generated in
  // C++ we instanciate a function in tablegen to map the valide configuration
  // to the corresponsding intrinsic ID.
  // Because we want a single source of truth, this mean the source of truth
  // about valid combinations needs to be in tablgen, therefore we generate
  // extra helpers to query valid configurations based on the shapes of
  // load/store operations.
  let extraClassDeclaration =
    "static llvm::Intrinsic::ID getIntrinsicID("
    "int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,"
    "mlir::NVVM::MMATypes eltypeEnum,mlir::NVVM::MMAFrag fragEnum) {"
    "llvm::StringRef layout = stringifyEnum(layoutEnum);"
    "llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    "llvm::StringRef frag = stringifyEnum(fragEnum);"
    #MMA_LD_INTR<"load">.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid n dimension based on mxk load shape.\n"
    "static int inferNDimension(int m, int k, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_N<!filter(op, NVVM_MMA_OPS.all_ld_ops, !eq(op.frag, "a"))>.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid m dimension based on kxn load shape.\n"
    "static int inferMDimension(int k, int n, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_M<!filter(op, NVVM_MMA_OPS.all_ld_ops, !eq(op.frag, "b"))>.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid k dimension based on mxn load shape.\n"
    "static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_K<!filter(op, NVVM_MMA_OPS.all_ld_ops, !eq(op.frag, "c"))>.id# "\n"
    "return 0;"
    "}\n";


  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = mlir::NVVM::WMMALoadOp::getIntrinsicID(
        $m, $n, $k, $layout, $eltype, $frag);
      $res = createIntrinsicCall(builder, intId, operands, {operands[0]->getType()});
  }];

  string baseDescription = [{
    The `nvvm.wmma.load` operation loads a matrix collectively using all the
    threads in a warp.

    The operation takes two arguments, the address from where the matrix
    elements are to be loaded from and a stride. The stride argument
    represents the leading dimension of the source matrix. The address and
    the stride are required to be the same across all threads in the warp.
    Each thread in a warp holds a certain number of elements. The Op returns
    a LLVMStruct which holds the elements of the matrix held by this thread.

    This op is meant to be used along with `nvvm.wmma.store` and
    `nvvm.wmma.mma`.

    Example:

    ```mlir
    %2 = nvvm.wmma.load %0, %1
      {eltype = "f16", frag = "a", k = 16 : i32, layout = "row", m = 16 : i32, n = 16 : i32}
      : (!llvm.ptr<i32, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    ```
    }];

  let assemblyFormat = "$ptr `,` $stride attr-dict `:` functional-type($ptr, $res)";
  let hasVerifier = 1;
}

def NVVM_WMMAStoreOp : NVVM_Op<"wmma.store">,
  Arguments<(ins LLVM_AnyPointer: $ptr,
             I32Attr:$m, I32Attr:$n, I32Attr:$k, MMALayoutAttr:$layout,
             MMATypesAttr:$eltype, Variadic<LLVM_Type>:$args, I32: $stride)>{
  let summary = "Warp synchronous matrix store";

  let extraClassDeclaration =
    "static llvm::Intrinsic::ID getIntrinsicID("
    "int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,"
    "mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef layout = stringifyEnum(layoutEnum);"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INTR<"store">.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid k dimension based on mxn store shape.\n"
    "static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_K<NVVM_MMA_OPS.all_st_ops>.id#  "\n"
    "return 0;"
    "}";

  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId =
        mlir::NVVM::WMMAStoreOp::getIntrinsicID($m, $n, $k, $layout, $eltype);
      createIntrinsicCall(builder, intId, operands, {operands[0]->getType()});
  }];

  string baseDescription = [{
    The `nvvm.wmma.store` operation stores a matrix collectively using
    all the threads in a warp.

    The operation takes as arguments the address to where the matrix elements are
    to be stored, a stride and the elements to store, held by the current thread.
    The stride argument represents the leading dimension of the destination matrix.
    The address and the stride are required to be the same across all threads in the
    warp.

    This op is meant to be used along with `nvvm.wmma.m16n16k16.load` and
    `nvvm.wmma.m16n16k16.mma`.

    Example:

    ```mlir
    nvvm.wmma.store %0, %1, %2, %3, %4, %5
      {eltype = "f16", k = 16 : i32, layout = "row", m = 16 : i32, n = 16 : i32}
      : !llvm.ptr<i32, 3>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
    ```
  }];

  let assemblyFormat = "$ptr `,` $stride `,` $args attr-dict `:` type($ptr) `,` type($args)";
  let hasVerifier = 1;
}

// Base class for all the variants of WMMA mmaOps that may be defined.
def NVVM_WMMAMmaOp : NVVM_Op<"wmma.mma">,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins I32Attr:$m, I32Attr:$n, I32Attr:$k, MMALayoutAttr:$layoutA,
             MMALayoutAttr:$layoutB, MMATypesAttr:$eltypeA,
             MMATypesAttr:$eltypeB, Variadic<LLVM_Type>:$args)>{
  let summary = "Warp synchronous matrix-multiply accumulate using tensor cores.";

  let extraClassDeclaration =
    "static llvm::Intrinsic::ID getIntrinsicID("
    "int m, int n, int k, mlir::NVVM::MMALayout layoutAEnum,"
    "mlir::NVVM::MMALayout layoutBEnum, mlir::NVVM::MMATypes eltypeAEnum,"
    "mlir::NVVM::MMATypes eltypeBEnum) {"
    "llvm::StringRef layoutA = stringifyEnum(layoutAEnum);"
    "llvm::StringRef layoutB = stringifyEnum(layoutBEnum);"
    "llvm::StringRef eltypeA = stringifyEnum(eltypeAEnum);"
    "llvm::StringRef eltypeB = stringifyEnum(eltypeBEnum);"
    #MMA_MMA_INTR<"mma">.id# "\n"
    "return 0;"
    "}";

  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = mlir::NVVM::WMMAMmaOp::getIntrinsicID(
        $m, $n, $k, $layoutA, $layoutB, $eltypeA, $eltypeB);
      $res = createIntrinsicCall(builder, intId, operands);
  }];

  string baseDescription = [{
    The `nvvm.wmma.mma` operation performs a matrix-multiply accumulate
    (mma) operation using all the threads in a warp.

    The operation performed is represented as `D = A * B + C`. The operation takes
    as arguments the elements of the matrices `A`, `B`, `C` and `D`, held by the
    current thread. The op returns a LLVM struct which holds a part of the result
    held by the current thread.

    This op is meant to be used along with `nvvm.wmma.load` and
    `nvvm.wmma.store`.

    Example:

    ```mlir
    %16 = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
      {eltypeA = "tf32", eltypeB = "f32", k = 8 : i32, layoutA = "row", layoutB = "row", m = 16 : i32, n = 16 : i32}
      : (i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32, f32, f32, f32)
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    ```
  }];

  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
  let hasVerifier = 1;
}

#endif // NVVMIR_OPS
