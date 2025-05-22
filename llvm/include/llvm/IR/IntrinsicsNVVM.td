//===- IntrinsicsNVVM.td - Defines NVVM intrinsics ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the NVVM-specific intrinsics for use with NVPTX.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Guidelines on NVPTX Intrinsic design
//===----------------------------------------------------------------------===//
// 
// The NVPTX intrinsics are used to model instructions in the PTX ISA.
// While simpler intrinsics can represent certain features effectively,
// more complex instructions like TMA and MMA are not as straightforward
// to model. A single variant of these complex instructions can expand
// into hundreds of intrinsics. Additionally, any expansion in the
// corresponding ISA can exponentially increase these numbers, making it
// difficult to manage them in the IR and backend passes. Therefore,
// a careful design of intrinsic interfaces can ease maintenance and
// contribute to a sustainable, long-term solution.
//
// The default approach is to have a 1:1 match between the intrinsic and
// the instruction where the instruction suffixes map to the intrinsic name
// and the instruction arguments map to the intrinsic arguments or return
// value.
//
// However, when there are too many instruction/intrinsic variants like
// the TMA/MMA family, it is desirable to encode some variants as a
// constant argument, referred to as 'flags'.
// TODO: Add a guideline to quantify the metric on 'how many intrinsics' here.
//
// Below are a set of guidelines that may help in choosing
// an appropriate design for the complex intrinsics:
// 
// 1. Each flag argument represents one set of instruction modifiers.
//    These flags are compile-time integer constants.
// 
// 2. When an intrinsic uses flags, document it with details of the
//    flag usage in the ``NVPTXUsage.rst`` file.
// 3. Annotate all flag arguments with ImmArg<ArgIdx<>>.
// 4. Place the flag arguments at the end of the (actual)argument list.
// 
// 5. Use `i1` for boolean flags and `i8` for others. Usually,
//    the `i8` types represent an `enum` encoding the family of
//    modifiers.
// 6. Note that, the specific variant for non-boolean flags may not be
//    obvious in the IR. So, maintain consistency between the enum value
//    definitions and their usage in the backend.
//    * Provide a meaningful default value in the enums wherever applicable.
//    * TODO: Investigate auto-upgrade capability for intrinsics
//      when only flag value mappings change.
//
// 7. Identify the key features of an intrinsic and distinguish between
//    first-order and supplementary information. Typically, encoding the
//    first-order information in the intrinsic name while using flags
//    for supplementary details improves readability.
//    For example:
// 
//    i. For MMA intrinsics, 'dense' vs. 'sparse' is a fundamental feature,
//    whereas an optional scaling applied to matrices is relatively secondary.
// 
//    ii. For TMAs, the mode of copy (e.g., 'Tile' or 'Im2col') is a first-order
//    information, while features like an optional cache hint tend to be
//    secondary.
// 
// 8. If there are invalid combinations within a set of modifiers, avoid
//    encoding them as flags, as much as possible. This helps reduce the
//    need for error handling of unsupported cases in the backend.
//    For example, some 'cvt' intrinsics support only a subset of the
//    possible rounding modes; so it is preferable not to encode the
//    rounding modes as flags.
// 9. Similarly, when there are invalid combinations across a set of
//    modifiers, avoid encoding them as flags to prevent additional
//    complexity in error handling.
// 
// 10. Maintain a consistent design within an intrinsic family, including
//     argument ordering as well as the usage and ordering of flags.
// 11. When designing an intrinsic corresponding to an instruction or its variant,
//     consider the entire instruction family. This may reveal common features
//     that can be modelled consistently across the family.
// 
// In summary, strive to balance the aspects mentioned above, to achieve
// a scalable design with maximum readability.
//===----------------------------------------------------------------------===//

// The following intrinsics were once defined here, but are now auto-upgraded
// to target-generic LLVM intrinsics.
//
//   * llvm.nvvm.brev32  --> llvm.bitreverse.i32
//   * llvm.nvvm.brev64  --> llvm.bitreverse.i64
//   * llvm.nvvm.clz.i   --> llvm.ctlz.i32
//   * llvm.nvvm.clz.ll  --> trunc i64 llvm.ctlz.i64(x) to i32
//   * llvm.nvvm.popc.i  --> llvm.ctpop.i32
//   * llvm.nvvm.popc.ll --> trunc i64 llvm.ctpop.i64 to i32
//   * llvm.nvvm.abs.i   --> select(x >= -x, x, -x)
//   * llvm.nvvm.abs.ll  --> ibid.
//   * llvm.nvvm.max.i   --> select(x sge y, x, y)
//   * llvm.nvvm.max.ll  --> ibid.
//   * llvm.nvvm.max.ui  --> select(x uge y, x, y)
//   * llvm.nvvm.max.ull --> ibid.
//   * llvm.nvvm.max.i   --> select(x sle y, x, y)
//   * llvm.nvvm.max.ll  --> ibid.
//   * llvm.nvvm.max.ui  --> select(x ule y, x, y)
//   * llvm.nvvm.max.ull --> ibid.
//   * llvm.nvvm.h2f     --> llvm.convert.to.fp16.f32
//   * llvm.nvvm.bitcast.f2i         --> bitcast
//   * llvm.nvvm.bitcast.i2f         --> ibid.
//   * llvm.nvvm.bitcast.d2ll        --> ibid.
//   * llvm.nvvm.bitcast.ll2d        --> ibid.
//   * llvm.nvvm.ptr.gen.to.global   --> addrspacecast
//   * llvm.nvvm.ptr.gen.to.shared   --> ibid.
//   * llvm.nvvm.ptr.gen.to.constant --> ibid.
//   * llvm.nvvm.ptr.gen.to.local    --> ibid.
//   * llvm.nvvm.ptr.gen.to.param    --> ibid.
//   * llvm.nvvm.ptr.global.to.gen   --> ibid.
//   * llvm.nvvm.ptr.shared.to.gen   --> ibid.
//   * llvm.nvvm.ptr.constant.to.gen --> ibid.
//   * llvm.nvvm.ptr.local.to.gen    --> ibid.
//   * llvm.nvvm.ptr.param.to.gen    --> ibid.
//   * llvm.nvvm.ldg.global.i        --> load addrspace(1) !load.invariant
//   * llvm.nvvm.ldg.global.f        --> ibid.
//   * llvm.nvvm.ldg.global.p        --> ibid.
//   * llvm.nvvm.swap.lo.hi.b64      --> llvm.fshl(x, x, 32)
//   * llvm.nvvm.atomic.load.inc.32  --> atomicrmw uinc_wrap
//   * llvm.nvvm.atomic.load.dec.32  --> atomicrmw udec_wrap
// * llvm.nvvm.barrier0              --> llvm.nvvm.barrier.cta.sync.aligned.all(0)
// * llvm.nvvm.barrier.n             --> llvm.nvvm.barrier.cta.sync.aligned.all(x)
// * llvm.nvvm.bar.sync              --> llvm.nvvm.barrier.cta.sync.aligned.all(x)
// * llvm.nvvm.barrier               --> llvm.nvvm.barrier.cta.sync.aligned(x, y)
// * llvm.nvvm.barrier.sync          --> llvm.nvvm.barrier.cta.sync.all(x)
// * llvm.nvvm.barrier.sync.cnt      --> llvm.nvvm.barrier.cta.sync(x, y)

def llvm_global_ptr_ty  : LLVMQualPointerType<1>;         // (global)ptr
def llvm_shared_ptr_ty  : LLVMQualPointerType<3>;         // (shared)ptr
def llvm_local_ptr_ty   : LLVMQualPointerType<5>;         // (local)ptr
def llvm_tmem_ptr_ty    : LLVMQualPointerType<6>;         // (tensor memory)ptr
def llvm_shared_cluster_ptr_ty : LLVMQualPointerType<7>;  // (shared_cluster)ptr

//
// MISC
//

defvar WARP_SIZE = 32;

// Note: the maximum grid size in the x-dimension is the lower value of 65535
// on sm_20. We conservatively use the larger value here as it required for
// sm_30+ and also correct for sm_20.
defvar MAX_GRID_SIZE_X = 0x7fffffff;
defvar MAX_GRID_SIZE_Y = 0xffff;
defvar MAX_GRID_SIZE_Z = 0xffff;

defvar MAX_BLOCK_SIZE_X = 1024;
defvar MAX_BLOCK_SIZE_Y = 1024;
defvar MAX_BLOCK_SIZE_Z = 64;

// Helper class that concatenates list elements with
// a given separator 'sep' and returns the result.
// Handles empty strings.
class StrJoin<string sep, list<string> str_list> {
  string ret = !foldl("", str_list, a, b,
               !if(!eq(a, ""), b, !if(!eq(b, ""), a, !strconcat(a, sep, b))));
}

// Helper class that represents a 'fragment' of an NVPTX *MMA instruction.
// Geom: m<M>n<N>k<K>. E.g. m8n32k16
// Frag: [a|b|c|d] ([x1|x2|x4] for ldmatrix)
// PtxEltType: PTX type for the element.
class WMMA_REGS<string Geom, string Frag, string PtxEltType> {
  string geom = Geom;
  string frag = Frag;
  string ptx_elt_type = PtxEltType;
  string gft = Geom#":"#Frag#":"#ptx_elt_type;
  string gf = Geom#":"#Frag;
  string ft = frag#":"#ptx_elt_type;
  list<LLVMType> regs = !cond(
    // mma fp ops use smaller fragments than wmma fp ops
    !eq(gft,"m8n8k4:a:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m8n8k4:b:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k8:a:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k8:b:f16") : [llvm_v2f16_ty],
    !eq(gft,"m16n8k8:c:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k8:d:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k8:c:f32") : !listsplat(llvm_float_ty, 4),
    !eq(gft,"m16n8k8:d:f32") : !listsplat(llvm_float_ty, 4),
    !eq(gft,"m16n8k16:a:f16") : !listsplat(llvm_v2f16_ty, 4),
    !eq(gft,"m16n8k16:b:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k16:c:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k16:d:f16") : !listsplat(llvm_v2f16_ty, 2),
    !eq(gft,"m16n8k16:c:f32") : !listsplat(llvm_float_ty, 4),
    !eq(gft,"m16n8k16:d:f32") : !listsplat(llvm_float_ty, 4),
    !eq(gft,"m16n8k4:c:f32") : !listsplat(llvm_float_ty, 4),
    !eq(gft,"m16n8k4:d:f32") : !listsplat(llvm_float_ty, 4),

    // wmma fp16 -> fp16/fp32 @  m16n16k16/m8n32k16/m32n8k16
    // All other supported geometries use the same fragment format for f32 and
    // f16, so we only need to consider {fragment, type}.
    !eq(ft,"a:f16") : !listsplat(llvm_v2f16_ty, 8),
    !eq(ft,"b:f16") : !listsplat(llvm_v2f16_ty, 8),
    !eq(ft,"c:f16") : !listsplat(llvm_v2f16_ty, 4),
    !eq(ft,"d:f16") : !listsplat(llvm_v2f16_ty, 4),
    !eq(ft,"c:f32") : !listsplat(llvm_float_ty, 8),
    !eq(ft,"d:f32") : !listsplat(llvm_float_ty, 8),

    // wmma tf32 -> s32 @ m16n16k8
    !eq(gft,"m16n16k8:a:tf32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n16k8:b:tf32") : !listsplat(llvm_i32_ty, 4),

    // mma tf32 -> s32 @ m16n16k8/m16n8k8
    !eq(gft,"m16n8k4:a:tf32") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k4:b:tf32") : [llvm_i32_ty],
    !eq(gft,"m16n8k8:a:tf32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k8:b:tf32") : !listsplat(llvm_i32_ty, 2),

    !eq(gft,"m8n8k4:a:f64") : [llvm_double_ty],
    !eq(gft,"m8n8k4:b:f64") : [llvm_double_ty],
    !eq(gft,"m8n8k4:c:f64") : !listsplat(llvm_double_ty, 2),
    !eq(gft,"m8n8k4:d:f64") : !listsplat(llvm_double_ty, 2),

    // wmma bf16 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    !eq(gft,"m16n16k16:a:bf16") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n16k16:b:bf16") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m8n32k16:a:bf16") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m8n32k16:b:bf16") : !listsplat(llvm_i32_ty, 8),
    !eq(gft,"m32n8k16:a:bf16") : !listsplat(llvm_i32_ty, 8),
    !eq(gft,"m32n8k16:b:bf16") : !listsplat(llvm_i32_ty, 2),

    // mma bf16 -> s32 @ m16n8k16/m16n8k8
    !eq(gft,"m16n8k16:a:bf16") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k16:b:bf16") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k8:a:bf16") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k8:b:bf16") : [llvm_i32_ty],

    // wmma u8/s8 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    !eq(gft,"m16n16k16:a:u8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n16k16:a:s8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n16k16:b:u8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n16k16:b:s8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n16k16:c:s32") : !listsplat(llvm_i32_ty, 8),
    !eq(gft,"m16n16k16:d:s32") : !listsplat(llvm_i32_ty, 8),

    !eq(gft,"m8n32k16:a:u8") : [llvm_i32_ty],
    !eq(gft,"m8n32k16:a:s8") : [llvm_i32_ty],
    !eq(gft,"m8n32k16:b:u8") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m8n32k16:b:s8") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m8n32k16:c:s32") : !listsplat(llvm_i32_ty, 8),
    !eq(gft,"m8n32k16:d:s32") : !listsplat(llvm_i32_ty, 8),

    !eq(gft,"m32n8k16:a:u8") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m32n8k16:a:s8") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m32n8k16:b:u8") : [llvm_i32_ty],
    !eq(gft,"m32n8k16:b:s8") : [llvm_i32_ty],
    !eq(gft,"m32n8k16:c:s32") : !listsplat(llvm_i32_ty, 8),
    !eq(gft,"m32n8k16:d:s32") : !listsplat(llvm_i32_ty, 8),

    // mma u8/s8 -> s32 @ m8n8k16/m16n8k16/m16n8k32
    !eq(gft,"m8n8k16:a:u8") : [llvm_i32_ty],
    !eq(gft,"m8n8k16:a:s8") : [llvm_i32_ty],
    !eq(gft,"m8n8k16:b:u8") : [llvm_i32_ty],
    !eq(gft,"m8n8k16:b:s8") : [llvm_i32_ty],
    !eq(gft,"m8n8k16:c:s32") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m8n8k16:d:s32") : !listsplat(llvm_i32_ty, 2),

    !eq(gft,"m16n8k16:a:u8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k16:a:s8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k16:b:u8") : [llvm_i32_ty],
    !eq(gft,"m16n8k16:b:s8") : [llvm_i32_ty],
    !eq(gft,"m16n8k16:c:s32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k16:d:s32") : !listsplat(llvm_i32_ty, 4),

    !eq(gft,"m16n8k32:a:u8") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k32:a:s8") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k32:b:u8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k32:b:s8") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k32:c:s32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k32:d:s32") : !listsplat(llvm_i32_ty, 4),

    // wmma/mma u4/s4 -> s32 @ m8n8k32 (u4/s4)
    !eq(gft,"m8n8k32:a:u4") : [llvm_i32_ty],
    !eq(gft,"m8n8k32:a:s4") : [llvm_i32_ty],
    !eq(gft,"m8n8k32:b:u4") : [llvm_i32_ty],
    !eq(gft,"m8n8k32:b:s4") : [llvm_i32_ty],
    !eq(gft,"m8n8k32:c:s32") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m8n8k32:d:s32") : !listsplat(llvm_i32_ty, 2),

    !eq(gft,"m16n8k32:a:u4") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k32:a:s4") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k32:b:u4") : [llvm_i32_ty],
    !eq(gft,"m16n8k32:b:s4") : [llvm_i32_ty],
    !eq(gft,"m16n8k32:c:s32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k32:d:s32") : !listsplat(llvm_i32_ty, 4),

    !eq(gft,"m16n8k64:a:u4") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k64:a:s4") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k64:b:u4") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k64:b:s4") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k64:c:s32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k64:d:s32") : !listsplat(llvm_i32_ty, 4),

    // wmma/mma b1 -> s32 @ m8n8k128(b1)
    !eq(gft,"m8n8k128:a:b1") : [llvm_i32_ty],
    !eq(gft,"m8n8k128:b:b1") : [llvm_i32_ty],
    !eq(gft,"m8n8k128:c:s32") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m8n8k128:d:s32") : !listsplat(llvm_i32_ty, 2),

    !eq(gft,"m16n8k128:a:b1") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k128:b:b1") : [llvm_i32_ty],
    !eq(gft,"m16n8k128:c:s32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k128:d:s32") : !listsplat(llvm_i32_ty, 4),

    !eq(gft,"m16n8k256:a:b1") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k256:b:b1") : !listsplat(llvm_i32_ty, 2),
    !eq(gft,"m16n8k256:c:s32") : !listsplat(llvm_i32_ty, 4),
    !eq(gft,"m16n8k256:d:s32") : !listsplat(llvm_i32_ty, 4),

    // ldmatrix b16 -> s32 @ m8n8
    !eq(gf,"m8n8:x1") : !listsplat(llvm_i32_ty, 1),
    !eq(gf,"m8n8:x2") : !listsplat(llvm_i32_ty, 2),
    !eq(gf,"m8n8:x4") : !listsplat(llvm_i32_ty, 4),

    // ldmatrix b8, b8x16.b6x16_p32, b8x16.b4x16_p64 -> s32 @ m16n16
    !eq(gf,"m16n16:x1") : !listsplat(llvm_i32_ty, 2),
    !eq(gf,"m16n16:x2") : !listsplat(llvm_i32_ty, 4),

    // ldmatrix b8x16.b6x16_p32, b8x16.b4x16_p64 -> s32 @ m8n16
    !eq(gf,"m8n16:x1") : !listsplat(llvm_i32_ty, 1),
    !eq(gf,"m8n16:x2") : !listsplat(llvm_i32_ty, 2),
    !eq(gf,"m8n16:x4") : !listsplat(llvm_i32_ty, 4),

  );
}

class WMMA_NAME_LDST<string Op, WMMA_REGS Frag, string Layout, int WithStride> {
  string intr = "llvm.nvvm.wmma."
                # Frag.geom
                # "." # Op
                # "." # Frag.frag
                # "." # Layout
                # !if(WithStride, ".stride", "")
                # "." # Frag.ptx_elt_type
                ;
  // TODO(tra): record name should ideally use the same field order as the intrinsic.
  // E.g. string record = !subst("llvm", "int",
  //                      !subst(".", "_", llvm));
  string record = "int_nvvm_wmma_"
                # Frag.geom
                # "_" # Op
                # "_" # Frag.frag
                # "_" # Frag.ptx_elt_type
                # "_" # Layout
                # !if(WithStride, "_stride", "");
}

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

class WMMA_NAME<string ALayout, string BLayout, int Satfinite, string Rnd, string b1op,
                WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  string signature = MMA_SIGNATURE<A, B, C, D>.ret;
  string record = "int_nvvm_wmma_"
                  # A.geom
                  # "_mma"
                  # !subst(".", "_", b1op)
                  # "_" # ALayout
                  # "_" # BLayout
                  # !if(!ne(Rnd, ""), !strconcat("_", Rnd), "")
                  # signature
                  # !if(Satfinite, "_satfinite", "");
}

class MMA_NAME<string ALayout, string BLayout, int Satfinite, string b1op,
               WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  string signature = MMA_SIGNATURE<A, B, C, D>.ret;
  string record = "int_nvvm_mma"
                  # !subst(".", "_", b1op)
                  # "_" # A.geom
                  # "_" # ALayout
                  # "_" # BLayout
                  # !if(Satfinite, "_satfinite", "")
                  # signature;
}

class LDMATRIX_NAME<WMMA_REGS Frag, int Trans> {
  string intr = "llvm.nvvm.ldmatrix.sync.aligned"
                # "." # Frag.geom
                # "." # Frag.frag
                # !if(Trans, ".trans", "")
                # "." # Frag.ptx_elt_type
                ;
  string record = !subst(".", "_",
                  !subst("llvm.", "int_", intr));
}

// Generates list of 4-tuples of WMMA_REGS representing a valid MMA op.
//   Geom: list of supported geometries.
//   TypeN: PTX type of the corresponding fragment's element.
//   TypeB and TypeD may be empty if it must match that of TypeA or TypeC.
class MMA_OPS<list<string> Geom, list<string> TypeA, list<string> TypeB,
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

class MMA_LDST_OPS<list<string> Geom, list<string> Frags, list<string> Types> {
  list<WMMA_REGS> ret =
     !foldl([]<WMMA_REGS>, Geom, t1, geom, !listconcat(t1,
     !foldl([]<WMMA_REGS>, Frags, t2, frag, !listconcat(t2,
     !foldl([]<WMMA_REGS>, Types, t3, type, !listconcat(t3,
            [WMMA_REGS<geom, frag, type>]))))));
   // Debugging aid for readable representation of the list above.
   list<string> ops = !foreach(x, ret, x.gft);
}

class LDMATRIX_OPS<list<string> Geom, list<string> Frags, list<string> Types> {
  list<WMMA_REGS> ret =
     !foldl([]<WMMA_REGS>, Geom, t1, geom, !listconcat(t1,
     !foldl([]<WMMA_REGS>, Frags, t2, frag, !listconcat(t2,
     !foldl([]<WMMA_REGS>, Types, t3, type, !listconcat(t3,
            [WMMA_REGS<geom, frag, type>]))))));
   // Debugging aid for readable representation of the list above.
   list<string> ops = !foreach(x, ret, x.gft);
}

// Creates list of valid combinations of fragments. This is the main list that
// drives generation of corresponding intrinsics and instructions.
class NVVM_MMA_OPS {
  list<list<WMMA_REGS>> tf32_wmma_ops = MMA_OPS<
            ["m16n16k8"],
            ["tf32"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> bf16_wmma_ops = MMA_OPS<
            ["m16n16k16", "m32n8k16", "m8n32k16"],
            ["bf16"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> f64_wmma_ops = MMA_OPS<
            ["m8n8k4"],
            ["f64"], [], ["f64"], []>.ret;
  list<list<WMMA_REGS>> fp_wmma_ops = MMA_OPS<
            ["m16n16k16", "m32n8k16", "m8n32k16"],
            ["f16"], [], ["f16", "f32"], ["f16", "f32"]>.ret;
  list<list<WMMA_REGS>> int_wmma_ops = MMA_OPS<
            ["m16n16k16", "m32n8k16", "m8n32k16"],
            ["s8", "u8"], [], ["s32"], []>.ret;
  list<list<WMMA_REGS>> subint_wmma_ops = MMA_OPS<
            ["m8n8k32"],
            ["s4", "u4"], [], ["s32"], []>.ret;
  list<list<WMMA_REGS>> bit_wmma_ops = MMA_OPS<
            ["m8n8k128"],
            ["b1"], [], ["s32"], []>.ret;
  list<list<WMMA_REGS>> all_wmma_ops = !listconcat(
            tf32_wmma_ops, bf16_wmma_ops, f64_wmma_ops,
            fp_wmma_ops, int_wmma_ops, subint_wmma_ops, bit_wmma_ops);

  list<list<WMMA_REGS>> tf32_mma_ops = MMA_OPS<
            ["m16n8k4", "m16n8k8"],
            ["tf32"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> bf16_mma_ops = MMA_OPS<
            ["m16n8k16", "m16n8k8"],
            ["bf16"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> f64_mma_ops = MMA_OPS<
            ["m8n8k4"],
            ["f64"], [], ["f64"], []>.ret;
  list<list<WMMA_REGS>> fp_mma_ops = MMA_OPS<
            ["m8n8k4", "m16n8k8", "m16n8k16"],
            ["f16"], [], ["f16", "f32"], ["f16", "f32"]>.ret;
  list<list<WMMA_REGS>> int_mma_ops = MMA_OPS<
            ["m8n8k16", "m16n8k16", "m16n8k32"],
            ["s8", "u8"], ["s8", "u8"], ["s32"], []>.ret;
  list<list<WMMA_REGS>> subint_mma_ops = MMA_OPS<
            ["m8n8k32", "m16n8k32", "m16n8k64"],
            ["s4", "u4"], ["s4", "u4"], ["s32"], []>.ret;
  list<list<WMMA_REGS>> bit_mma_ops = MMA_OPS<
            ["m8n8k128", "m16n8k128", "m16n8k256"],
            ["b1"], [], ["s32"], []>.ret;
  list<list<WMMA_REGS>> all_mma_ops = !listconcat(
            tf32_mma_ops, bf16_mma_ops, f64_mma_ops,
            fp_mma_ops, int_mma_ops, subint_mma_ops, bit_mma_ops);

  list<WMMA_REGS> ldst_ab_ops = MMA_LDST_OPS<
            ["m16n16k16", "m32n8k16", "m8n32k16"],
            ["a", "b"], ["f16", "u8", "s8", "bf16"]>.ret;
  list<WMMA_REGS> ldst_cd_ops = MMA_LDST_OPS<
            ["m16n16k16", "m32n8k16", "m8n32k16"],
            ["c", "d"], ["f16", "f32", "s32"]>.ret;
  list<WMMA_REGS> ldst_tf32_ab_ops = MMA_LDST_OPS<
            ["m16n16k8"],
            ["a", "b"], ["tf32"]>.ret;
  list<WMMA_REGS> ldst_tf32_cd_ops = MMA_LDST_OPS<
            ["m16n16k8"],
            ["c", "d"], ["f32"]>.ret;
  list<WMMA_REGS> ldst_f64_abcd_ops = MMA_LDST_OPS<
            ["m8n8k4"],
            ["a", "b", "c", "d"], ["f64"]>.ret;
  list<WMMA_REGS> ldst_subint_ab_ops = MMA_LDST_OPS<
            ["m8n8k32"], ["a", "b"], ["s4","u4"]>.ret;
  list<WMMA_REGS> ldst_bit_ab_ops = MMA_LDST_OPS<
            ["m8n8k128"], ["a", "b"], ["b1"]>.ret;
  list<WMMA_REGS> ldst_subint_cd_ops = MMA_LDST_OPS<
            ["m8n8k32", "m8n8k128"],  ["c", "d"], ["s32"]>.ret;
  list<WMMA_REGS> all_ldst_ops = !listconcat(ldst_ab_ops, ldst_cd_ops,
                                             ldst_tf32_ab_ops,
                                             ldst_tf32_cd_ops,
                                             ldst_f64_abcd_ops,
                                             ldst_subint_ab_ops,
                                             ldst_bit_ab_ops,
                                             ldst_subint_cd_ops);
  // Separate A/B/C fragments (loads) from D (stores).
  list<WMMA_REGS> all_ld_ops = !filter(op, all_ldst_ops, !ne(op.frag, "d"));
  list<WMMA_REGS> all_st_ops = !filter(op, all_ldst_ops, !eq(op.frag, "d"));

  list<WMMA_REGS> ldmatrix_b16_ops = LDMATRIX_OPS<
    ["m8n8"], ["x1", "x2", "x4"], ["b16"]>.ret;

  list<WMMA_REGS> ldmatrix_geom_m16n16_ops = LDMATRIX_OPS<
    ["m16n16"], ["x1", "x2"], ["b8", "b8x16.b6x16_p32", "b8x16.b4x16_p64"]>.ret;

  list<WMMA_REGS> ldmatrix_geom_m8n16_ops = LDMATRIX_OPS<
    ["m8n16"], ["x1", "x2", "x4"], ["b8x16.b6x16_p32", "b8x16.b4x16_p64"]>.ret;

  list<WMMA_REGS> all_ldmatrix_ops = !listconcat(ldmatrix_b16_ops,
                                                 ldmatrix_geom_m16n16_ops,
                                                 ldmatrix_geom_m8n16_ops);
}

def NVVM_MMA_OPS : NVVM_MMA_OPS;

// Returns true if this combination of fragment and layout for WMMA load/store
// ops is supported; false otherwise.
// E.g.
// if NVVM_WMMA_LDST_SUPPORTED<...>.ret then
//   def : FOO<>; // The record will only be defined for supported ops.
//
class NVVM_WMMA_LDST_SUPPORTED<WMMA_REGS frag, string layout> {
  string f = frag.frag;
  string t = frag.ptx_elt_type;

  bit ret = !cond(
    // Sub-int load and store requires A fragment to be of row layout and B
    // fragments to be of column layout.
    !and(!or(!eq(t, "b1"),
             !eq(t, "u4"),
             !eq(t, "s4")),
         !or(!and(!eq(f, "a"),
                  !ne(layout, "row")),
             !and(!eq(f, "b"),
                  !ne(layout, "col")))) : false,
    true: true
  );
}

// Returns true if this combination of layout/satf/rnd for WMMA ops is
// supported; false otherwise.
// E.g.
// if NVVM_WMMA_SUPPORTED<...>.ret then
//   def : FOO<>; // The record will only be defined for supported ops.
//
class NVVM_WMMA_SUPPORTED<list<WMMA_REGS> frags, string layout_a, string layout_b, int satf, string rnd> {
  // WMMA ops check both layouts.
  string layout = layout_a # ":" # layout_b;
  string t = frags[0].ptx_elt_type;

  bit ret = !cond(
    // only f64 wmma functions support rnd options
    // any non f64 type that uses a rnd value is invalid
    !and(!ne(t, "f64"), !ne(rnd, "")) : false,

    // satf is only valid for select types
    !and(!eq(satf, 1),
         !ne(t, "s8"),
         !ne(t, "u8"),
         !ne(t, "s4"),
         !ne(t, "u4"),
         !ne(t, "f16")): false,

    // Sub-int wmma requires row/column layout
    !and(!or(!eq(t, "s4"),
             !eq(t, "u4"),
             !eq(t, "b1")),
         !ne(layout, "row:col")) : false,
    true: true
  );
}

class NVVM_MMA_B1OPS<list<WMMA_REGS> frags> {
  list<string> ret = !cond(
    !eq(frags[0].ptx_elt_type, "b1") : [".xor.popc", ".and.popc"],
    true: [""]
  );
}

// Returns true if this combination of layout/satf for MMA ops is supported;
// false otherwise.
// E.g.
// if NVVM_MMA_SUPPORTED<...>.ret then
//   def : FOO<>; // The record will only be defined for supported ops.
//
class NVVM_MMA_SUPPORTED<list<WMMA_REGS> frags, string layout_a, string layout_b, int satf> {
  // MMA ops check both layouts.
  string layout = layout_a # ":" # layout_b;
  string a_type = frags[0].ptx_elt_type;
  string b_type = frags[1].ptx_elt_type;
  string c_type = frags[2].ptx_elt_type;
  string d_type = frags[3].ptx_elt_type;
  string geom = frags[0].geom;

  // gcd is a shortcut used to identify instructions that depend on
  // geom+frag_c+frag_d.
  string gcd = geom # ":" # c_type # d_type;
  bit ret = !cond(

    // Limit satf to valid types
    !and(!eq(satf, 1),
         !ne(a_type, "s8"),
         !ne(a_type, "u8"),
         !ne(a_type, "s4"),
         !ne(a_type, "u4")): false,

    // m8n8k4 has no C=f32 D=f16 variant.
    !eq(gcd, "m8n8k4:f32f16"): false,

    // only m8n8k4 for f16 does not require row:col layout
    !and(!ne(layout, "row:col"),
         !or(!ne(geom, "m8n8k4"),
             !ne(a_type, "f16"))) : false,

    // m16n8k8 requires A and B to be the same type and C and D to be the same
    // type.
    !and(!eq(geom, "m16n8k8"),
         !or(!ne(a_type, b_type),
             !ne(c_type, d_type))): false,

    // m16n8k8 requires C and D to be the same type.
    !and(!eq(geom, "m16n8k8"),
         !ne(c_type, d_type)): false,

    // All other are OK.
    true: true
  );
}

// Returns true if the fragment is valid for ldmatrix ops is supported;
// false otherwise.
// E.g.
// if NVVM_LDMATRIX_SUPPORTED<...>.ret then
//   def : FOO<>; // The record will only be defined for supported ops.
//
class NVVM_LDMATRIX_SUPPORTED<WMMA_REGS frag, bit trans> {
  string g = frag.geom;
  string t = frag.ptx_elt_type;

  bit ret = !cond(
    !and(!eq(g, "m8n8"), !eq(t, "b16")): true,
    !and(!eq(g, "m16n16"), !eq(t, "b8"), !eq(trans, 1)): true,
    !and(!eq(g, "m16n16"), !eq(t, "b8x16.b6x16_p32"), !eq(trans, 1)): true,
    !and(!eq(g, "m16n16"), !eq(t, "b8x16.b4x16_p64"), !eq(trans, 1)): true,
    !and(!eq(g, "m8n16"), !eq(t, "b8"), !eq(trans, 0)): true,
    !and(!eq(g, "m8n16"), !eq(t, "b8x16.b6x16_p32"), !eq(trans, 0)): true,
    !and(!eq(g, "m8n16"), !eq(t, "b8x16.b4x16_p64"), !eq(trans, 0)): true,
    true: false
  );
}

class SHFL_INFO<bit sync, string mode, string type, bit return_pred> {
  string Suffix = !if(sync, "sync_", "")
                  # mode # "_"
                  # type
                  # !if(return_pred, "p", "");

  string Name = "int_nvvm_shfl_" # Suffix;
  bit withGccBuiltin = !not(return_pred);
  LLVMType OpType = !cond(
    !eq(type,"i32"): llvm_i32_ty,
    !eq(type,"f32"): llvm_float_ty);
  list<LLVMType> RetTy = !if(return_pred, [OpType, llvm_i1_ty], [OpType]);
  list<LLVMType> ArgsTy = !if(sync,
    [llvm_i32_ty, OpType, llvm_i32_ty, llvm_i32_ty],
    [OpType, llvm_i32_ty, llvm_i32_ty]);
}

class NVVM_TCGEN05_LDST_ACCESS_SIZE<string Shape, int Num> {
  int shift = !cond(!eq(Shape, "16x128b"): 1,
                    !eq(Shape, "16x256b"): 2,
                    true : 0);

  int veclen = !shl(1, !add(Num, shift));

  int valid = !le(veclen, 128);
  LLVMType type = !cond(!eq(veclen,   1): llvm_i32_ty,
                        !eq(veclen,   2): llvm_v2i32_ty,
                        !eq(veclen,   4): llvm_v4i32_ty,
                        !eq(veclen,   8): llvm_v8i32_ty,
                        !eq(veclen,  16): llvm_v16i32_ty,
                        !eq(veclen,  32): llvm_v32i32_ty,
                        !eq(veclen,  64): llvm_v64i32_ty,
                        !eq(veclen, 128): llvm_v128i32_ty,
                        true : llvm_void_ty);
}

class TexVector<string name, list<LLVMType> types> {
  string Name = name;
  list<LLVMType> Types = types;
}

def TV_I8 : TexVector<"i8", [llvm_i16_ty]>;
def TV_I16 : TexVector<"i16", [llvm_i16_ty]>;
def TV_I32 : TexVector<"i32", [llvm_i32_ty]>;
def TV_I64 : TexVector<"i64", [llvm_i64_ty]>;
def TV_V2I8 : TexVector<"v2i8", !listsplat(llvm_i16_ty, 2)>;
def TV_V2I16 : TexVector<"v2i16", !listsplat(llvm_i16_ty, 2)>;
def TV_V2I32 : TexVector<"v2i32", !listsplat(llvm_i32_ty, 2)>;
def TV_V2I64 : TexVector<"v2i64", !listsplat(llvm_i64_ty, 2)>;
def TV_V4I8 : TexVector<"v4i8", !listsplat(llvm_i16_ty, 4)>;
def TV_V4I16 : TexVector<"v4i16", !listsplat(llvm_i16_ty, 4)>;
def TV_V4I32 : TexVector<"v4i32", !listsplat(llvm_i32_ty, 4)>;


def V4F32 : TexVector<"v4f32", !listsplat(llvm_float_ty, 4)>;
def V4S32 : TexVector<"v4s32", !listsplat(llvm_i32_ty, 4)>;
def V4U32 : TexVector<"v4u32", !listsplat(llvm_i32_ty, 4)>;

class NVVMBuiltin :
  ClangBuiltin<!strconcat("__", !substr(NAME, !size("int_")))> {
    assert !eq(!substr(NAME, 0, !size("int_nvvm_")), "int_nvvm_"),
           "NVVMBuiltin must be a NVVM intrinsic starting with 'int_nvvm_'";
}

let TargetPrefix = "nvvm" in {
  def int_nvvm_prmt : NVVMBuiltin,
      DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, IntrSpeculatable]>;

  def int_nvvm_nanosleep : NVVMBuiltin,
      DefaultAttrsIntrinsic<[], [llvm_i32_ty],
                            [IntrConvergent, IntrNoMem, IntrHasSideEffects]>;

//
// Min Max
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable, Commutative] in {
    foreach operation = ["min", "max"] in {
      def int_nvvm_f # operation # _d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty]>;

      foreach variant = ["", "_xorsign_abs"] in {
        foreach nan = ["", "_nan"] in {
          foreach ftz = ["", "_ftz"] in {
            def int_nvvm_f # operation # ftz # nan # variant # _f : NVVMBuiltin,
              DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]>;

            def int_nvvm_f # operation # ftz # nan # variant # _f16 :
              DefaultAttrsIntrinsic<[llvm_half_ty], [llvm_half_ty, llvm_half_ty]>;

            def int_nvvm_f # operation # ftz # nan # variant # _f16x2 :
              DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_v2f16_ty, llvm_v2f16_ty]>;

            def int_nvvm_f # operation # ftz # nan # variant # _bf16 : NVVMBuiltin,
              DefaultAttrsIntrinsic<[llvm_bfloat_ty], [llvm_bfloat_ty, llvm_bfloat_ty]>;

            def int_nvvm_f # operation # ftz # nan # variant # _bf16x2 : NVVMBuiltin,
              DefaultAttrsIntrinsic<[llvm_v2bf16_ty], [llvm_v2bf16_ty, llvm_v2bf16_ty]>;
          } // ftz
        } // nan
      } // variant
    } // operation
  }

//
// Multiplication
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable, Commutative] in {
    foreach sign = ["", "u"] in {
      def int_nvvm_mulhi_ # sign # s : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty]>;

      def int_nvvm_mulhi_ # sign # i : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;

      def int_nvvm_mulhi_ # sign # ll : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty]>;

      def int_nvvm_mul24_ # sign # i : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;
    }

    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_mul_ # rnd # ftz # _f : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]>;

      def int_nvvm_mul_ # rnd # _d : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty]>;
    }
  }

//
// Div
//
  let IntrProperties = [IntrNoMem] in {
    foreach ftz = ["", "_ftz"] in {
      def int_nvvm_div_approx # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]>;

      def int_nvvm_div_full # ftz : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]>;
    }

    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_div_ # rnd # ftz # _f : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]>;

      def int_nvvm_div_ # rnd # _d : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty]>;
    }
  }

//
// Sad
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    foreach sign = ["", "u"] in {
      def int_nvvm_sad_ # sign # s : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty, llvm_i16_ty]>;

      def int_nvvm_sad_ # sign # i : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;

      def int_nvvm_sad_ # sign # ll : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty]>;
    }
  }

//
// Floor  Ceil
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    foreach op = ["floor", "ceil"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_ # op # ftz # _f : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;
      def int_nvvm_ # op # _d : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
    }
  }

//
// Abs
//
  foreach ftz = ["", "_ftz"] in
    def int_nvvm_fabs # ftz :
      DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>],
                            [IntrNoMem, IntrSpeculatable]>;

//
// Abs, Neg bf16, bf16x2
//
  def int_nvvm_neg_bf16 : NVVMBuiltin,
    DefaultAttrsIntrinsic<[llvm_bfloat_ty], [llvm_bfloat_ty], [IntrNoMem]>;
  def int_nvvm_neg_bf16x2 : NVVMBuiltin,
    DefaultAttrsIntrinsic<[llvm_v2bf16_ty], [llvm_v2bf16_ty], [IntrNoMem]>;

//
// Round
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    foreach ftz = ["", "_ftz"] in
      def int_nvvm_round # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

    def int_nvvm_round_d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
  }

//
// Trunc
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    foreach ftz = ["", "_ftz"] in
      def int_nvvm_trunc # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

    def int_nvvm_trunc_d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
  }

//
// Saturate
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    foreach ftz = ["", "_ftz"] in
      def int_nvvm_saturate # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

    def int_nvvm_saturate_d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
  }

//
// Exp2  Log2
//
  let IntrProperties = [IntrNoMem] in {
    foreach ftz = ["", "_ftz"] in
      def int_nvvm_ex2_approx # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

    def int_nvvm_ex2_approx_d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
    def int_nvvm_ex2_approx_f16 :
        DefaultAttrsIntrinsic<[llvm_half_ty], [llvm_half_ty]>;
    def int_nvvm_ex2_approx_f16x2 :
        DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_v2f16_ty]>;

    foreach ftz = ["", "_ftz"] in
      def int_nvvm_lg2_approx # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

    def int_nvvm_lg2_approx_d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
  }

//
// Sin  Cos
//
  foreach op = ["sin", "cos"] in
    foreach ftz = ["", "_ftz"] in
      def int_nvvm_ # op # _approx # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

//
// Fma
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    foreach variant = ["", "_sat", "_relu"] in {
      foreach ftz = ["", "_ftz"] in {
        def int_nvvm_fma_rn # ftz # variant # _f16 :
          DefaultAttrsIntrinsic<[llvm_half_ty],
            [llvm_half_ty, llvm_half_ty, llvm_half_ty]>;

        def int_nvvm_fma_rn # ftz # variant # _f16x2 :
          DefaultAttrsIntrinsic<[llvm_v2f16_ty],
            [llvm_v2f16_ty, llvm_v2f16_ty, llvm_v2f16_ty]>;

        def int_nvvm_fma_rn # ftz # variant # _bf16 : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_bfloat_ty],
            [llvm_bfloat_ty, llvm_bfloat_ty, llvm_bfloat_ty]>;

        def int_nvvm_fma_rn # ftz # variant # _bf16x2 : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_v2bf16_ty],
            [llvm_v2bf16_ty, llvm_v2bf16_ty, llvm_v2bf16_ty]>;
      } // ftz
    } // variant

    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_fma_ # rnd # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_float_ty, llvm_float_ty]>;

      def int_nvvm_fma_ # rnd # _d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty],
          [llvm_double_ty, llvm_double_ty, llvm_double_ty]>;
    }
  }

//
// Rcp
//
  let IntrProperties = [IntrNoMem] in {
    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_rcp_ # rnd # ftz # _f : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

      def int_nvvm_rcp_ # rnd # _d : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
    }

    def int_nvvm_rcp_approx_ftz_f : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;
    def int_nvvm_rcp_approx_ftz_d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
  }

//
// Sqrt
//
  let IntrProperties = [IntrNoMem] in {
    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_sqrt_ # rnd # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

      def int_nvvm_sqrt_ # rnd # _d : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
    }

    def int_nvvm_sqrt_f : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;

    foreach ftz = ["", "_ftz"] in
      def int_nvvm_sqrt_approx # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;
  }

//
// Rsqrt
//
  let IntrProperties = [IntrNoMem] in {
    foreach ftz = ["", "_ftz"] in {
      def int_nvvm_rsqrt_approx # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;
      def int_nvvm_rsqrt_approx # ftz # _d : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
    }
  }

//
// Add
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable, Commutative] in {
    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_add_ # rnd # ftz # _f : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]>;

    def int_nvvm_add_ # rnd # _d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty]>;
    }
  }

//
// Dot Product
//
  foreach a_type = ["s", "u"] in {
    foreach b_type = ["s", "u"] in {
      def int_nvvm_idp4a_ # a_type # _ # b_type :
          DefaultAttrsIntrinsic<[llvm_i32_ty],
              [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrSpeculatable]>;
      def int_nvvm_idp2a_ # a_type # _ # b_type :
          DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;
    }
  }

//
// Funnel-shift
//
  foreach direction = ["l", "r"] in
    def int_nvvm_fsh # direction # _clamp :
      DefaultAttrsIntrinsic<[llvm_anyint_ty],
        [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
        [IntrNoMem, IntrSpeculatable]>;

//
// FLO - Find Leading One
//
  foreach sign = ["s", "u"] in
    def int_nvvm_flo_ # sign :
      DefaultAttrsIntrinsic<[llvm_i32_ty],
        [llvm_anyint_ty, llvm_i1_ty],
        [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<1>>]>;

//
// szext
//
  foreach ext = ["sext", "zext"] in
    foreach mode = ["wrap", "clamp"] in
      def int_nvvm_ # ext # _ # mode :
        DefaultAttrsIntrinsic<[llvm_i32_ty],
          [llvm_i32_ty, llvm_i32_ty],
          [IntrNoMem, IntrSpeculatable]>;

//
// BMSK - bit mask
//
  foreach mode = ["wrap", "clamp"] in
    def int_nvvm_bmsk_ # mode :
      DefaultAttrsIntrinsic<[llvm_i32_ty],
        [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, IntrSpeculatable]>;

//
// Convert
//
  let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
    def int_nvvm_lohi_i2d : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_i32_ty, llvm_i32_ty]>;

    def int_nvvm_d2i_lo : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_double_ty]>;
    def int_nvvm_d2i_hi : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_double_ty]>;

    foreach rnd = ["rn", "rz", "rm", "rp"] in {
      foreach ftz = ["", "_ftz"] in
        def int_nvvm_d2f_ # rnd # ftz : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_double_ty]>;

      foreach sign = ["", "u"] in {

        def int_nvvm_d2 # sign # i_ # rnd : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_double_ty]>;

        def int_nvvm_ # sign # i2d_ # rnd : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_i32_ty]>;

        foreach ftz = ["", "_ftz"] in
          def int_nvvm_f2 # sign # i_ # rnd # ftz : NVVMBuiltin,
              DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_float_ty]>;

        def int_nvvm_ # sign # i2f_ # rnd : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_i32_ty]>;

        foreach ftz = ["", "_ftz"] in
          def int_nvvm_f2 # sign # ll_ # rnd # ftz : NVVMBuiltin,
              DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_float_ty]>;

        def int_nvvm_d2 # sign # ll_ # rnd : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_double_ty]>;

        def int_nvvm_ # sign # ll2f_ # rnd : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_i64_ty]>;

        def int_nvvm_ # sign # ll2d_ # rnd : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_i64_ty]>;

      } // sign
    } // rnd

    foreach ftz = ["", "_ftz"] in {
      def int_nvvm_f2h_rn # ftz : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_float_ty]>;

      def int_nvvm_bf2h_rn # ftz : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_bfloat_ty]>;
    }

    foreach rnd = ["rn", "rz"] in {
      foreach relu = ["", "_relu"] in {
        def int_nvvm_ff2bf16x2_ # rnd # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_v2bf16_ty], [llvm_float_ty, llvm_float_ty]>;

        def int_nvvm_ff2f16x2_ # rnd # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_float_ty, llvm_float_ty]>;

        def int_nvvm_f2bf16_ # rnd # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_bfloat_ty], [llvm_float_ty]>;
      }
    }

    foreach satfinite = ["", "_satfinite"] in {
      def int_nvvm_f2tf32_rna # satfinite : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_float_ty]>;

      foreach rnd = ["rn", "rz"] in
        foreach relu = ["", "_relu"] in
          def int_nvvm_f2tf32_ # rnd # relu # satfinite : NVVMBuiltin,
              DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_float_ty]>;
    }

    foreach type = ["e4m3x2", "e5m2x2"] in {
      foreach relu = ["", "_relu"] in {
        def int_nvvm_ff_to_ # type # _rn # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_float_ty, llvm_float_ty]>;

        def int_nvvm_f16x2_to_ # type # _rn # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_v2f16_ty]>;

        def int_nvvm_ # type # _to_f16x2_rn # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_i16_ty]>;
      }
    }

    // FP4 conversions.
    foreach relu = ["", "_relu"] in {
      def int_nvvm_ff_to_e2m1x2_rn # relu # _satfinite : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_float_ty, llvm_float_ty]>;

      def int_nvvm_e2m1x2_to_f16x2_rn # relu : NVVMBuiltin,
          DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_i16_ty]>;
    }

    // FP6 conversions.
    foreach type = ["e2m3x2", "e3m2x2"] in {
      foreach relu = ["", "_relu"] in {
        def int_nvvm_ff_to_ # type # _rn # relu # _satfinite : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_float_ty, llvm_float_ty]>;

        def int_nvvm_ # type # _to_f16x2_rn # relu : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_i16_ty]>;
      }
    }

    // UE8M0x2 conversions.
    foreach rmode = ["_rz", "_rp"] in {
      foreach satmode = ["", "_satfinite"] in {
        defvar suffix = rmode # satmode;
        def int_nvvm_ff_to_ue8m0x2 # suffix : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_float_ty, llvm_float_ty]>;

        def int_nvvm_bf16x2_to_ue8m0x2 # suffix : NVVMBuiltin,
            DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_v2bf16_ty]>;

      }
    }

    def int_nvvm_ue8m0x2_to_bf16x2 : NVVMBuiltin,
        DefaultAttrsIntrinsic<[llvm_v2bf16_ty], [llvm_i16_ty]>;

  } // IntrProperties = [IntrNoMem, IntrSpeculatable]

// FNS
  def int_nvvm_fns : NVVMBuiltin,
      DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem]>;

  class SCOPED_ATOMIC2_impl<LLVMType elty>
        : Intrinsic<[elty],
          [llvm_anyptr_ty, LLVMMatchType<0>],
          [IntrArgMemOnly, IntrNoCallback, NoCapture<ArgIndex<0>>]>;
  class SCOPED_ATOMIC3_impl<LLVMType elty>
        : Intrinsic<[elty],
          [llvm_anyptr_ty, LLVMMatchType<0>, LLVMMatchType<0>],
          [IntrArgMemOnly, IntrNoCallback, NoCapture<ArgIndex<0>>]>;

  multiclass PTXAtomicWithScope2<LLVMType elty> {
    def _cta : SCOPED_ATOMIC2_impl<elty>;
    def _sys : SCOPED_ATOMIC2_impl<elty>;
  }
  multiclass PTXAtomicWithScope3<LLVMType elty> {
    def _cta : SCOPED_ATOMIC3_impl<elty>;
    def _sys : SCOPED_ATOMIC3_impl<elty>;
  }
  multiclass PTXAtomicWithScope2_fi {
    defm _f : PTXAtomicWithScope2<llvm_anyfloat_ty>;
    defm _i : PTXAtomicWithScope2<llvm_anyint_ty>;
  }
  defm int_nvvm_atomic_add_gen    : PTXAtomicWithScope2_fi;
  defm int_nvvm_atomic_inc_gen_i  : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_dec_gen_i  : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_exch_gen_i : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_xor_gen_i  : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_max_gen_i  : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_min_gen_i  : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_or_gen_i   : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_and_gen_i  : PTXAtomicWithScope2<llvm_anyint_ty>;
  defm int_nvvm_atomic_cas_gen_i  : PTXAtomicWithScope3<llvm_anyint_ty>;

// Bar.Sync
  def int_nvvm_barrier0_popc : ClangBuiltin<"__nvvm_bar0_popc">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;
  def int_nvvm_barrier0_and : ClangBuiltin<"__nvvm_bar0_and">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;
  def int_nvvm_barrier0_or : ClangBuiltin<"__nvvm_bar0_or">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;

  def int_nvvm_bar_warp_sync : NVVMBuiltin,
      Intrinsic<[], [llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;

  // barrier{.cta}.sync{.aligned}      a{, b};
  // barrier{.cta}.arrive{.aligned}    a, b;
  let IntrProperties = [IntrConvergent, IntrNoCallback] in {
    foreach align = ["", "_aligned"] in {
      def int_nvvm_barrier_cta_sync # align # _all :
          Intrinsic<[], [llvm_i32_ty]>;
      def int_nvvm_barrier_cta_sync # align :
          Intrinsic<[], [llvm_i32_ty, llvm_i32_ty]>;
      def int_nvvm_barrier_cta_arrive # align :
          Intrinsic<[], [llvm_i32_ty, llvm_i32_ty]>;
    }
  }

  // barrier.cluster.[wait, arrive, arrive.relaxed]
  def int_nvvm_barrier_cluster_arrive :
      Intrinsic<[], [], [IntrConvergent, IntrNoCallback]>;
  def int_nvvm_barrier_cluster_arrive_relaxed :
      Intrinsic<[], [], [IntrConvergent, IntrNoCallback]>;
  def int_nvvm_barrier_cluster_wait :
      Intrinsic<[], [], [IntrConvergent, IntrNoCallback]>;

  // 'aligned' versions of the above barrier.cluster.* intrinsics
  def int_nvvm_barrier_cluster_arrive_aligned :
      Intrinsic<[], [], [IntrConvergent, IntrNoCallback]>;
  def int_nvvm_barrier_cluster_arrive_relaxed_aligned :
      Intrinsic<[], [], [IntrConvergent, IntrNoCallback]>;
  def int_nvvm_barrier_cluster_wait_aligned :
      Intrinsic<[], [], [IntrConvergent, IntrNoCallback]>;

  // Membar
  def int_nvvm_membar_cta : NVVMBuiltin, Intrinsic<[], [], [IntrNoCallback]>;
  def int_nvvm_membar_gl : NVVMBuiltin, Intrinsic<[], [], [IntrNoCallback]>;
  def int_nvvm_membar_sys : NVVMBuiltin, Intrinsic<[], [], [IntrNoCallback]>;
  def int_nvvm_fence_sc_cluster : Intrinsic<[], [], [IntrNoCallback]>;

// Proxy fence (uni-directional)
foreach scope = ["cta", "cluster", "gpu", "sys"] in {

  def int_nvvm_fence_proxy_tensormap_generic_release_ # scope :
        Intrinsic<[], [], [IntrNoCallback],
        "llvm.nvvm.fence.proxy.tensormap_generic.release." # scope>;

  def int_nvvm_fence_proxy_tensormap_generic_acquire_ # scope :
        Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty],
                  [IntrNoCallback, IntrArgMemOnly, ImmArg<ArgIndex<1>>],
                  "llvm.nvvm.fence.proxy.tensormap_generic.acquire." # scope>;
}

// Async Copy
let IntrProperties = [IntrConvergent, IntrNoCallback] in {
  def int_nvvm_cp_async_mbarrier_arrive : NVVMBuiltin,
      Intrinsic<[],[llvm_ptr_ty]>;
  def int_nvvm_cp_async_mbarrier_arrive_shared : NVVMBuiltin,
      Intrinsic<[],[llvm_shared_ptr_ty]>;
  def int_nvvm_cp_async_mbarrier_arrive_noinc : NVVMBuiltin,
      Intrinsic<[],[llvm_ptr_ty]>;
  def int_nvvm_cp_async_mbarrier_arrive_noinc_shared : NVVMBuiltin,
      Intrinsic<[],[llvm_shared_ptr_ty]>;
}

multiclass CP_ASYNC_SHARED_GLOBAL {
  def NAME : Intrinsic<[], [llvm_shared_ptr_ty, llvm_global_ptr_ty],
        [IntrArgMemOnly, IntrNoCallback, NoAlias<ArgIndex<0>>, NoAlias<ArgIndex<1>>,
        WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>]>;
  def _s : Intrinsic<[], [llvm_shared_ptr_ty, llvm_global_ptr_ty, llvm_i32_ty],
        [IntrArgMemOnly, IntrNoCallback, NoAlias<ArgIndex<0>>, NoAlias<ArgIndex<1>>,
        WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>]>;
}

defm int_nvvm_cp_async_ca_shared_global_4 : CP_ASYNC_SHARED_GLOBAL;
defm int_nvvm_cp_async_ca_shared_global_8 : CP_ASYNC_SHARED_GLOBAL;
defm int_nvvm_cp_async_ca_shared_global_16 : CP_ASYNC_SHARED_GLOBAL;
defm int_nvvm_cp_async_cg_shared_global_16 : CP_ASYNC_SHARED_GLOBAL;

def int_nvvm_cp_async_commit_group : NVVMBuiltin, Intrinsic<[], [], []>;

def int_nvvm_cp_async_wait_group : NVVMBuiltin,
    Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_nvvm_cp_async_wait_all : NVVMBuiltin,
    Intrinsic<[], [], []>;

// cp.async.bulk variants of the commit/wait group
def int_nvvm_cp_async_bulk_commit_group :
    Intrinsic<[], [], []>;

def int_nvvm_cp_async_bulk_wait_group :
    Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_nvvm_cp_async_bulk_wait_group_read :
    Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

// mbarrier
def int_nvvm_mbarrier_init : NVVMBuiltin,
    Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_init_shared : NVVMBuiltin,
    Intrinsic<[], [llvm_shared_ptr_ty, llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;

def int_nvvm_mbarrier_inval : NVVMBuiltin,
    Intrinsic<[], [llvm_ptr_ty],
    [IntrConvergent, IntrWriteMem, IntrArgMemOnly, IntrNoCallback,
    WriteOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>]>;
def int_nvvm_mbarrier_inval_shared : NVVMBuiltin,
    Intrinsic<[], [llvm_shared_ptr_ty],
    [IntrConvergent, IntrWriteMem, IntrArgMemOnly, IntrNoCallback,
    WriteOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>]>;

def int_nvvm_mbarrier_arrive : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_ptr_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_arrive_shared : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_shared_ptr_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_arrive_noComplete : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_arrive_noComplete_shared : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_shared_ptr_ty,
    llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;

def int_nvvm_mbarrier_arrive_drop : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_ptr_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_arrive_drop_shared : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_shared_ptr_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_arrive_drop_noComplete : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_arrive_drop_noComplete_shared : NVVMBuiltin,
    Intrinsic<[llvm_i64_ty], [llvm_shared_ptr_ty, llvm_i32_ty], [IntrConvergent, IntrNoCallback]>;

def int_nvvm_mbarrier_test_wait : NVVMBuiltin,
    Intrinsic<[llvm_i1_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrConvergent, IntrNoCallback]>;
def int_nvvm_mbarrier_test_wait_shared : NVVMBuiltin,
    Intrinsic<[llvm_i1_ty], [llvm_shared_ptr_ty, llvm_i64_ty], [IntrConvergent, IntrNoCallback]>;

def int_nvvm_mbarrier_pending_count : NVVMBuiltin,
    Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrConvergent, IntrNoCallback]>;

// Generated within nvvm. Use for ldu on sm_20 or later.  Second arg is the
// pointer's alignment.
let IntrProperties = [IntrReadMem, IntrArgMemOnly, IntrNoCallback, IntrWillReturn, NoCapture<ArgIndex<0>>] in {
  def int_nvvm_ldu_global_i : Intrinsic<[llvm_anyint_ty], [llvm_anyptr_ty, llvm_i32_ty]>;
  def int_nvvm_ldu_global_f : Intrinsic<[llvm_anyfloat_ty], [llvm_anyptr_ty, llvm_i32_ty]>;
  def int_nvvm_ldu_global_p : Intrinsic<[llvm_anyptr_ty], [llvm_anyptr_ty, llvm_i32_ty]>;
}

// Represents an explicit hole in the LLVM IR type system. It may be inserted by
// the compiler in cases where a pointer is of the wrong type. In the backend
// this intrinsic will be folded away and not equate to any instruction. It
// should not be used by any frontend and should only be considered well defined
// when added in the following cases:
//
//  - NVPTXLowerArgs: When wrapping a byval pointer argument to a kernel
//    function to convert the address space from generic (0) to param (101).
//    This accounts for the fact that the parameter symbols will occupy this
//    space when lowered during ISel.
//
def int_nvvm_internal_addrspace_wrap :
  DefaultAttrsIntrinsic<[llvm_anyptr_ty], [llvm_anyptr_ty],
                        [IntrNoMem, IntrSpeculatable, NoUndef<ArgIndex<0>>,
                         NoUndef<RetIndex>]>;

// Move intrinsics, used in nvvm internally

let IntrProperties = [IntrNoMem] in {
  def int_nvvm_move_i16 : DefaultAttrsIntrinsic<[llvm_i16_ty], [llvm_i16_ty]>;
  def int_nvvm_move_i32 : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty]>;
  def int_nvvm_move_i64 : DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty]>;
  def int_nvvm_move_float : DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty]>;
  def int_nvvm_move_double : DefaultAttrsIntrinsic<[llvm_double_ty], [llvm_double_ty]>;
  def int_nvvm_move_ptr : DefaultAttrsIntrinsic<[llvm_anyptr_ty], [llvm_anyptr_ty]>;
}

// For getting the handle from a texture or surface variable
let IntrProperties = [IntrNoMem, IntrSpeculatable] in {
  def int_nvvm_texsurf_handle
    : DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_metadata_ty, llvm_anyptr_ty]>;
  def int_nvvm_texsurf_handle_internal
    : DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_anyptr_ty]>;
}

/// Error / Warn
def int_nvvm_compiler_error : Intrinsic<[], [llvm_anyptr_ty]>;
def int_nvvm_compiler_warn : Intrinsic<[], [llvm_anyptr_ty]>;

def int_nvvm_reflect : NVVMBuiltin,
  Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;

// isspacep.{const, global, local, shared}
foreach space = ["const", "global", "local", "shared", "shared_cluster"] in
  def int_nvvm_isspacep_ # space : NVVMBuiltin,
    DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_ptr_ty],
              [IntrNoMem, IntrSpeculatable, NoCapture<ArgIndex<0>>]>;

// Environment register read
foreach i = 0...31 in
  def int_nvvm_read_ptx_sreg_envreg # i : NVVMBuiltin,
    DefaultAttrsIntrinsic<[llvm_i32_ty], [],
              [IntrNoMem, IntrSpeculatable, NoUndef<RetIndex>]>;

//
// Texture Fetch
//
let IntrProperties = [IntrReadMem] in {
  foreach is_unified = [true, false] in {
    defvar mode = !if(is_unified, "_unified", "");
    defvar addr_args = !if(is_unified, [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty]);

    foreach vec = [V4F32, V4S32, V4U32] in {
      foreach is_array = [true, false] in {
        defvar array = !if(is_array, "_array", "");
        defvar array_args = !if(is_array, [llvm_i32_ty], []<LLVMType>);

        def int_nvvm_tex # mode # _1d # array # _ # vec.Name # _s32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_i32_ty, 1))>;
        def int_nvvm_tex # mode # _1d # array # _ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 1))>;
        def int_nvvm_tex # mode # _1d # array # _level_ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 2))>;
        def int_nvvm_tex # mode # _1d # array # _grad_ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 3))>;

        def int_nvvm_tex # mode # _2d # array # _ # vec.Name # _s32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_i32_ty, 2))>;
        def int_nvvm_tex # mode # _2d # array # _ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 2))>;
        def int_nvvm_tex # mode # _2d # array # _level_ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 3))>;
        def int_nvvm_tex # mode # _2d # array # _grad_ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 6))>;

        if !not(is_array) then {
          def int_nvvm_tex # mode # _3d_ # vec.Name # _s32
            : Intrinsic<vec.Types,
                        !listconcat(addr_args, !listsplat(llvm_i32_ty, 3))>;
          def int_nvvm_tex # mode # _3d_ # vec.Name # _f32
            : Intrinsic<vec.Types,
                        !listconcat(addr_args, !listsplat(llvm_float_ty, 3))>;
          def int_nvvm_tex # mode # _3d_level_ # vec.Name # _f32
            : Intrinsic<vec.Types,
                        !listconcat(addr_args, !listsplat(llvm_float_ty, 4))>;
          def int_nvvm_tex # mode # _3d_grad_ # vec.Name # _f32
            : Intrinsic<vec.Types,
                        !listconcat(addr_args, !listsplat(llvm_float_ty, 9))>;
        }

        def int_nvvm_tex # mode # _cube # array # _ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 3))>;
        def int_nvvm_tex # mode # _cube # array # _level_ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 4))>;

        if is_unified then
          def int_nvvm_tex # mode # _cube # array # _grad_ # vec.Name # _f32
            : Intrinsic<vec.Types,
                        !listconcat(addr_args, array_args, !listsplat(llvm_float_ty, 9))>;
      } // is_array

      foreach comp = ["r", "g", "b", "a"] in {
        def int_nvvm_tld4 # mode # _ # comp # _2d_ # vec.Name # _f32
          : Intrinsic<vec.Types,
                      !listconcat(addr_args, !listsplat(llvm_float_ty, 2))>;
      } // comp
    } // vec
  } // is_unified
} // IntrProperties = [IntrReadMem]

//=== Surface Load
let IntrProperties = [IntrReadMem] in {
  foreach clamp = ["clamp", "trap", "zero"] in {
    foreach vec = [TV_I8, TV_I16, TV_I32, TV_I64,
                  TV_V2I8, TV_V2I16, TV_V2I32, TV_V2I64,
                  TV_V4I8, TV_V4I16, TV_V4I32] in {

      def int_nvvm_suld_1d_ # vec.Name # _ # clamp
        : Intrinsic<vec.Types,
                    [llvm_i64_ty, llvm_i32_ty]>;

      def int_nvvm_suld_1d_array_ # vec.Name # _ # clamp
        : Intrinsic<vec.Types,
                    [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty]>;

      def int_nvvm_suld_2d_ # vec.Name # _ # clamp
        : Intrinsic<vec.Types,
                    [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty]>;

      def int_nvvm_suld_2d_array_ # vec.Name # _ # clamp
        : Intrinsic<vec.Types,
                    [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;

      def int_nvvm_suld_3d_ # vec.Name # _ # clamp
        : Intrinsic<vec.Types,
                    [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;
    } // vec
  } // clamp
} // IntrProperties = [IntrReadMem]

//===- Texture Query ------------------------------------------------------===//

foreach query = ["channel_order", "channel_data_type", "width", "height",
                 "depth", "array_size", "num_samples", "num_mipmap_levels"] in
  def int_nvvm_txq_ # query : NVVMBuiltin,
    Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem]>;

//===- Surface Query ------------------------------------------------------===//

foreach query = ["channel_order", "channel_data_type", "width", "height",
                 "depth", "array_size"] in
  def int_nvvm_suq_ # query : NVVMBuiltin,
    Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem]>;

//===- Handle Query -------------------------------------------------------===//

foreach type = ["sampler", "surface", "texture"] in
  def int_nvvm_istypep_ # type : NVVMBuiltin,
    Intrinsic<[llvm_i1_ty], [llvm_i64_ty], [IntrNoMem]>;

//===- Surface Stores -----------------------------------------------------===//

multiclass SurfaceStoreIntrinsics<string clamp, TexVector vec> {
  def _1d_ # vec.Name # _ # clamp : NVVMBuiltin,
      Intrinsic<[], !listconcat([llvm_i64_ty, llvm_i32_ty], vec.Types)>;

  def _1d_array_ # vec.Name # _ # clamp : NVVMBuiltin,
      Intrinsic<[], !listconcat([llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], vec.Types)>;

  def _2d_ # vec.Name # _ # clamp : NVVMBuiltin,
      Intrinsic<[], !listconcat([llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], vec.Types)>;

  def _2d_array_ # vec.Name # _ # clamp : NVVMBuiltin,
      Intrinsic<[], !listconcat([llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], vec.Types)>;

  def _3d_ # vec.Name # _ # clamp : NVVMBuiltin,
      Intrinsic<[], !listconcat([llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], vec.Types)>;
}

// Unformatted
foreach clamp = ["clamp", "trap", "zero"] in
  foreach vec = [TV_I8, TV_I16, TV_I32, TV_I64,
                 TV_V2I8, TV_V2I16, TV_V2I32, TV_V2I64,
                 TV_V4I8, TV_V4I16, TV_V4I32] in
    defm int_nvvm_sust_b : SurfaceStoreIntrinsics<clamp, vec>;

// Formatted
foreach vec = [TV_I8, TV_I16, TV_I32,
               TV_V2I8, TV_V2I16, TV_V2I32,
               TV_V4I8, TV_V4I16, TV_V4I32] in
  defm int_nvvm_sust_p : SurfaceStoreIntrinsics<"trap", vec>;

// Accessing special registers.

class PTXReadSRegIntrinsicNB_r32<list<IntrinsicProperty> properties = []>
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [],
      !listconcat([IntrNoMem, IntrSpeculatable, NoUndef<RetIndex>], properties)>;

class PTXReadSRegIntrinsic_r32<list<IntrinsicProperty> properties = []>
  : PTXReadSRegIntrinsicNB_r32<properties>,
    NVVMBuiltin;

multiclass PTXReadSRegIntrinsic_v4i32<list<list<IntrinsicProperty>> properties = [[], [], [], []]> {
  assert !eq(!size(properties), 4), "properties must be a list of 4 lists";
// FIXME: Do we need the 128-bit integer type version?
//    def _r64   : Intrinsic<[llvm_i128_ty],   [], [IntrNoMem, IntrSpeculatable]>;

// FIXME: Enable this once v4i32 support is enabled in back-end.
//    def _v4i16 : Intrinsic<[llvm_v4i32_ty], [], [IntrNoMem, IntrSpeculatable]>;
  defvar suffixes = ["_x", "_y", "_z", "_w"];
  foreach i = !range(suffixes) in
    def suffixes[i] : PTXReadSRegIntrinsic_r32<properties[i]>;
}

// Same, but without automatic clang builtins. It will be used for
// registers that require particular GPU or PTX version.
multiclass PTXReadSRegIntrinsicNB_v4i32<list<list<IntrinsicProperty>> properties = [[], [], [], []]> {
  assert !eq(!size(properties), 4), "properties must be a list of 4 lists";
  defvar suffixes = ["_x", "_y", "_z", "_w"];
  foreach i = !range(suffixes) in
    def suffixes[i] : PTXReadSRegIntrinsicNB_r32<properties[i]>;
}

// Intrinsics to read registers with non-constant values. E.g. the values that
// do change over the kernel lifetime. Such reads should not be CSE'd.
class PTXReadNCSRegIntrinsic_r32
  : Intrinsic<[llvm_i32_ty], [], [IntrInaccessibleMemOnly, IntrNoCallback, NoUndef<RetIndex>]>,
    NVVMBuiltin;
class PTXReadNCSRegIntrinsic_r64
  : Intrinsic<[llvm_i64_ty], [], [IntrInaccessibleMemOnly, IntrNoCallback, NoUndef<RetIndex>]>,
    NVVMBuiltin;

defm int_nvvm_read_ptx_sreg_tid
  : PTXReadSRegIntrinsic_v4i32<[[Range<RetIndex, 0, MAX_BLOCK_SIZE_X>],
                                [Range<RetIndex, 0, MAX_BLOCK_SIZE_Y>],
                                [Range<RetIndex, 0, MAX_BLOCK_SIZE_Z>],
                                [Range<RetIndex, 0, 1>]]>;

defm int_nvvm_read_ptx_sreg_ntid
  : PTXReadSRegIntrinsic_v4i32<[[Range<RetIndex, 1, !add(MAX_BLOCK_SIZE_X, 1)>],
                                [Range<RetIndex, 1, !add(MAX_BLOCK_SIZE_Y, 1)>],
                                [Range<RetIndex, 1, !add(MAX_BLOCK_SIZE_Z, 1)>],
                                [Range<RetIndex, 0, 1>]]>;

def int_nvvm_read_ptx_sreg_laneid
  : PTXReadSRegIntrinsic_r32<[Range<RetIndex, 0, WARP_SIZE>]>;

def int_nvvm_read_ptx_sreg_warpid : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_nwarpid : PTXReadSRegIntrinsic_r32;

defvar MAX_GRID_ID_RANGE = [[Range<RetIndex, 0, MAX_GRID_SIZE_X>],
                            [Range<RetIndex, 0, MAX_GRID_SIZE_Y>],
                            [Range<RetIndex, 0, MAX_GRID_SIZE_Z>],
                            [Range<RetIndex, 0, 1>]];

defvar MAX_GRID_NID_RANGE = [[Range<RetIndex, 1, !add(MAX_GRID_SIZE_X, 1)>],
                             [Range<RetIndex, 1, !add(MAX_GRID_SIZE_Y, 1)>],
                             [Range<RetIndex, 1, !add(MAX_GRID_SIZE_Z, 1)>],
                             [Range<RetIndex, 0, 1>]];

defm int_nvvm_read_ptx_sreg_ctaid
  : PTXReadSRegIntrinsic_v4i32<MAX_GRID_ID_RANGE>;

defm int_nvvm_read_ptx_sreg_nctaid
  : PTXReadSRegIntrinsic_v4i32<MAX_GRID_NID_RANGE>;

def int_nvvm_read_ptx_sreg_smid : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_nsmid : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_gridid : PTXReadSRegIntrinsic_r32;

def int_nvvm_read_ptx_sreg_lanemask_eq : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_lanemask_le : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_lanemask_lt : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_lanemask_ge : PTXReadSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_lanemask_gt : PTXReadSRegIntrinsic_r32;

def int_nvvm_read_ptx_sreg_clock : PTXReadNCSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_clock64 : PTXReadNCSRegIntrinsic_r64;

def int_nvvm_read_ptx_sreg_globaltimer : PTXReadNCSRegIntrinsic_r64;

def int_nvvm_read_ptx_sreg_pm0 : PTXReadNCSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_pm1 : PTXReadNCSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_pm2 : PTXReadNCSRegIntrinsic_r32;
def int_nvvm_read_ptx_sreg_pm3 : PTXReadNCSRegIntrinsic_r32;

def int_nvvm_read_ptx_sreg_warpsize
  : PTXReadSRegIntrinsic_r32<[Range<RetIndex, WARP_SIZE, !add(WARP_SIZE, 1)>]>;

// sm90+, PTX7.8+

// Note: Since clusters are subdivisions of the grid, we conservatively use the
// maximum grid size as an upper bound for the clusterid and cluster_ctaid. In
// practice, the clusterid will likely be much smaller. The CUDA programming
// guide recommends 8 as a maximum portable value and H100s support 16.

defm int_nvvm_read_ptx_sreg_clusterid
  : PTXReadSRegIntrinsicNB_v4i32<MAX_GRID_ID_RANGE>;
defm int_nvvm_read_ptx_sreg_nclusterid
  : PTXReadSRegIntrinsicNB_v4i32<MAX_GRID_NID_RANGE>;
defm int_nvvm_read_ptx_sreg_cluster_ctaid
  : PTXReadSRegIntrinsicNB_v4i32<MAX_GRID_ID_RANGE>;
defm int_nvvm_read_ptx_sreg_cluster_nctaid
  : PTXReadSRegIntrinsicNB_v4i32<MAX_GRID_NID_RANGE>;

def int_nvvm_read_ptx_sreg_cluster_ctarank : PTXReadSRegIntrinsicNB_r32;
def int_nvvm_read_ptx_sreg_cluster_nctarank : PTXReadSRegIntrinsicNB_r32;

//
// SHUFFLE
//
// Generate intrinsics for all variants of shfl instruction.
let IntrProperties = [IntrInaccessibleMemOnly, IntrConvergent, IntrNoCallback] in {
  foreach sync = [false, true] in {
    foreach mode = ["up", "down", "bfly", "idx"] in {
      foreach type = ["i32", "f32"] in {
        foreach return_pred = [false, true] in {
          defvar i = SHFL_INFO<sync, mode, type, return_pred>;
          if i.withGccBuiltin then
            def i.Name : NVVMBuiltin, Intrinsic<i.RetTy, i.ArgsTy>;
          else
            def i.Name : Intrinsic<i.RetTy, i.ArgsTy>;
        }
      }
    }
  }
}

//
// VOTE
//

let IntrProperties = [IntrInaccessibleMemOnly, IntrConvergent, IntrNoCallback] in {
  def int_nvvm_vote_all : NVVMBuiltin, Intrinsic<[llvm_i1_ty], [llvm_i1_ty]>;
  def int_nvvm_vote_any : NVVMBuiltin, Intrinsic<[llvm_i1_ty], [llvm_i1_ty]>;
  def int_nvvm_vote_uni : NVVMBuiltin, Intrinsic<[llvm_i1_ty], [llvm_i1_ty]>;
  def int_nvvm_vote_ballot : NVVMBuiltin, Intrinsic<[llvm_i32_ty], [llvm_i1_ty]>;
}
//
// VOTE.SYNC
//
let IntrProperties = [IntrInaccessibleMemOnly, IntrConvergent, IntrNoCallback] in {
  def int_nvvm_vote_all_sync : NVVMBuiltin, Intrinsic<[llvm_i1_ty], [llvm_i32_ty, llvm_i1_ty]>;
  def int_nvvm_vote_any_sync : NVVMBuiltin, Intrinsic<[llvm_i1_ty], [llvm_i32_ty, llvm_i1_ty]>;
  def int_nvvm_vote_uni_sync : NVVMBuiltin, Intrinsic<[llvm_i1_ty], [llvm_i32_ty, llvm_i1_ty]>;
  def int_nvvm_vote_ballot_sync : NVVMBuiltin, Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i1_ty]>;
}

//
// ACTIVEMASK
//
def int_nvvm_activemask : NVVMBuiltin,
  Intrinsic<[llvm_i32_ty], [],
            [IntrInaccessibleMemOnly, IntrConvergent, IntrNoCallback, IntrHasSideEffects]>;

//
// MATCH.SYNC
//
let IntrProperties = [IntrInaccessibleMemOnly, IntrConvergent, IntrNoCallback] in {
  // match.any.sync.b32 mask, value
  def int_nvvm_match_any_sync_i32 : NVVMBuiltin,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;
  // match.any.sync.b64 mask, value
  def int_nvvm_match_any_sync_i64 : NVVMBuiltin,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty]>;

  // match.all instruction have two variants -- one returns a single value, another
  // returns a pair {value, predicate}. We currently only implement the latter as
  // that's the variant exposed by CUDA API.

  // match.all.sync.b32p mask, value
  def int_nvvm_match_all_sync_i32p :
    Intrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty]>;
  // match.all.sync.b64p mask, value
  def int_nvvm_match_all_sync_i64p :
    Intrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty, llvm_i64_ty]>;
}
//
// ELECT.SYNC
//
// elect.sync dst|pred, membermask
def int_nvvm_elect_sync :
  DefaultAttrsIntrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty],
                        [IntrInaccessibleMemOnly, IntrConvergent]>;

//
// REDUX.SYNC
//
// redux.sync.op.u32 dst, src, membermask;

let IntrProperties = [IntrConvergent, IntrInaccessibleMemOnly, IntrNoCallback] in {
  foreach op = ["umin", "umax", "add", "min", "max", "and", "xor", "or"] in
    def int_nvvm_redux_sync_ # op : NVVMBuiltin,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;

  // redux.sync.op.{abs}.{NaN}.f32 dst, src, membermask;
  foreach binOp = ["min", "max"] in
    foreach abs = ["", "_abs"] in
      foreach NaN = ["", "_NaN"] in
        def int_nvvm_redux_sync_f # binOp # abs # NaN : NVVMBuiltin,
          Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_i32_ty]>;
}

//
// WGMMA fence instructions
//
// wgmma.fence.sync.aligned;
def int_nvvm_wgmma_fence_sync_aligned : Intrinsic<[], [], [IntrConvergent]>;

// wgmma.commit_group.sync.aligned;
def int_nvvm_wgmma_commit_group_sync_aligned
  : Intrinsic<[], [], [IntrConvergent], "llvm.nvvm.wgmma.commit_group.sync.aligned">;

// wgmma.wait_group.sync.aligned N;
def int_nvvm_wgmma_wait_group_sync_aligned
  : Intrinsic<[], [llvm_i64_ty], [IntrConvergent, ImmArg<ArgIndex<0>>], "llvm.nvvm.wgmma.wait_group.sync.aligned">;

//
// WMMA instructions
//
// WMMA.LOAD
class NVVM_WMMA_LD<WMMA_REGS Frag, string Layout, int WithStride>
  : Intrinsic<Frag.regs,
              !if(WithStride, [llvm_anyptr_ty, llvm_i32_ty], [llvm_anyptr_ty]),
              [IntrWillReturn, IntrReadMem, IntrArgMemOnly, IntrNoCallback, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>],
              WMMA_NAME_LDST<"load", Frag, Layout, WithStride>.intr>;

// WMMA.STORE.D
class NVVM_WMMA_ST<WMMA_REGS Frag, string Layout, int WithStride>
  : Intrinsic<[],
              !listconcat(
                [llvm_anyptr_ty],
                Frag.regs,
                !if(WithStride, [llvm_i32_ty], [])),
              [IntrWriteMem, IntrArgMemOnly, IntrNoCallback, WriteOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>],
              WMMA_NAME_LDST<"store", Frag, Layout, WithStride>.intr>;

// Create all load/store variants
foreach layout = ["row", "col"] in {
  foreach stride = [0, 1] in {
    foreach frag = NVVM_MMA_OPS.all_ld_ops in
      if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
        def WMMA_NAME_LDST<"load", frag, layout, stride>.record
             : NVVM_WMMA_LD<frag, layout, stride>;
    foreach frag = NVVM_MMA_OPS.all_st_ops in
      if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
        def WMMA_NAME_LDST<"store", frag, layout, stride>.record
             : NVVM_WMMA_ST<frag, layout, stride>;
  }
}

// WMMA.MMA
class NVVM_MMA<WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D>
  : Intrinsic<D.regs,
              !listconcat(A.regs, B.regs, C.regs),
              [IntrNoMem, IntrNoCallback]>;

foreach layout_a = ["row", "col"] in {
  foreach layout_b = ["row", "col"] in {
    foreach satf = [0, 1] in {
      foreach rnd = ["", "rn", "rz", "rm", "rp"] in {
        foreach op = NVVM_MMA_OPS.all_wmma_ops in {
          foreach b1op = NVVM_MMA_B1OPS<op>.ret in {
            if NVVM_WMMA_SUPPORTED<op, layout_a, layout_b, satf, rnd>.ret then {
              def WMMA_NAME<layout_a, layout_b, satf, rnd, b1op,
                                op[0], op[1], op[2], op[3]>.record
                : NVVM_MMA<op[0], op[1], op[2], op[3]>;
            }
          } // b1op
        } // op
      } // rnd
    } // satf
  } // layout_b
} // layout_a

foreach layout_a = ["row", "col"] in {
  foreach layout_b = ["row", "col"] in {
    foreach satf = [0, 1] in {
      foreach op = NVVM_MMA_OPS.all_mma_ops in {
        foreach b1op = NVVM_MMA_B1OPS<op>.ret in {
          if NVVM_MMA_SUPPORTED<op, layout_a, layout_b, satf>.ret then {
            def MMA_NAME<layout_a, layout_b, satf, b1op, op[0], op[1], op[2], op[3]>.record
              : NVVM_MMA<op[0], op[1], op[2], op[3]>;
          }
        } // b1op
      } // op
    } // satf
  } // layout_b
} // layout_a

// LDMATRIX
class NVVM_LDMATRIX<WMMA_REGS Frag, int Transposed>
  : Intrinsic<Frag.regs, [llvm_anyptr_ty],
              [IntrReadMem, IntrArgMemOnly, IntrNoCallback, ReadOnly<ArgIndex<0>>,
               NoCapture<ArgIndex<0>>],
              LDMATRIX_NAME<Frag, Transposed>.intr>;

foreach transposed = [0, 1] in {
  foreach frag = NVVM_MMA_OPS.all_ldmatrix_ops in {
    if NVVM_LDMATRIX_SUPPORTED<frag, transposed>.ret then {
      def LDMATRIX_NAME<frag, transposed>.record
        : NVVM_LDMATRIX<frag, transposed>;
    }
  }
}

// MAPA
let IntrProperties = [IntrNoMem, IntrSpeculatable, NoCapture<ArgIndex<0>>] in {
  def int_nvvm_mapa
    : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty]>;
  def int_nvvm_mapa_shared_cluster
    : DefaultAttrsIntrinsic<[llvm_shared_cluster_ptr_ty], [llvm_shared_ptr_ty, llvm_i32_ty]>;
}

// GETCTARANK
let IntrProperties = [IntrNoMem, IntrSpeculatable, NoCapture<ArgIndex<0>>] in {
  def int_nvvm_getctarank
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_ptr_ty]>;
  def int_nvvm_getctarank_shared_cluster
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_shared_ptr_ty]>;
}

def int_nvvm_is_explicit_cluster
  : DefaultAttrsIntrinsic<[llvm_i1_ty], [],
              [IntrNoMem, IntrSpeculatable, NoUndef<RetIndex>],
              "llvm.nvvm.is_explicit_cluster">;

// Setmaxnreg inc/dec intrinsics
foreach op = ["dec", "inc"] in
  def int_nvvm_setmaxnreg_ # op # _sync_aligned_u32
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty],
              [IntrConvergent, IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// Exit
def int_nvvm_exit : NVVMBuiltin,
    Intrinsic<[], [], [IntrConvergent, IntrInaccessibleMemOnly, IntrNoReturn]>;

class DefaultAttrsIntrinsicFlags<list<LLVMType> ret_types,
                list<LLVMType> param_types,
                list<LLVMType> flags,
                list<IntrinsicProperty> intr_properties>
  : DefaultAttrsIntrinsic<
        ret_types,
        !listconcat(param_types, flags),
        !listconcat(intr_properties,
                    !foreach(i, !range(flags),
                        ImmArg<ArgIndex<!add(i, !size(param_types))>>))>;

// Intrinsics for Tensor Copy using TMA
// G2S -> From Global to Shared memory variants
// S2G -> From Shared to Global memory variants
foreach dim = 1...5 in {
  defvar tensor_dim_args = !listsplat(llvm_i32_ty, dim);

  foreach mode = !if(!ge(dim, 3), ["tile", "im2col"], ["tile"]) in {
    defvar is_im2col = !eq(mode, "im2col");
    defvar num_im2col_offsets = !if(is_im2col, !add(dim, -2), 0);
    defvar im2col_offsets_args = !listsplat(llvm_i16_ty, num_im2col_offsets);

    def int_nvvm_cp_async_bulk_tensor_g2s_ # mode # _ # dim # d :
      DefaultAttrsIntrinsicFlags<[],
          !listconcat([llvm_shared_cluster_ptr_ty,  // dst_shared_cluster_ptr
                       llvm_shared_ptr_ty,          // mbarrier_smem_ptr
                       llvm_ptr_ty],                // tensormap_ptr
                      tensor_dim_args,              // actual tensor dims
                      im2col_offsets_args,          // im2col offsets
                      [llvm_i16_ty,                 // cta_mask
                       llvm_i64_ty]),               // cache_hint
          [llvm_i1_ty,                              // Flag for cta_mask
           llvm_i1_ty],                             // Flag for cache_hint
          [IntrConvergent,
           WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<2>>,
           NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>, NoCapture<ArgIndex<2>>]>;

    def int_nvvm_cp_async_bulk_tensor_s2g_ # mode # _ # dim # d :
      DefaultAttrsIntrinsicFlags<[],
          !listconcat([llvm_shared_ptr_ty,  // src_smem_ptr
                       llvm_ptr_ty],        // tensormap_ptr
                      tensor_dim_args,      // actual tensor dims
                      [llvm_i64_ty]),       // cache_hint
          [llvm_i1_ty],                     // Flag for cache_hint
          [IntrConvergent,
           ReadOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>,
           NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>]>;

    def int_nvvm_cp_async_bulk_tensor_prefetch_ # mode # _ # dim # d :
      DefaultAttrsIntrinsicFlags<[],
          !listconcat([llvm_ptr_ty],        // tensormap_ptr
                       tensor_dim_args,     // actual tensor dims
                       im2col_offsets_args, // im2col offsets
                      [llvm_i64_ty]),       // cache_hint
          [llvm_i1_ty],                     // Flag for cache_hint
          [IntrConvergent,
           ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>]>;

    // Intrinsics for TMA Copy with reduction
    foreach red_op = ["add", "min", "max", "inc", "dec", "and", "or", "xor"] in
      def int_nvvm_cp_async_bulk_tensor_reduce_ # red_op # _ # mode # _ # dim # d :
        DefaultAttrsIntrinsicFlags<[],
            !listconcat([llvm_shared_ptr_ty,  // src_smem_ptr
                         llvm_ptr_ty],        // tensormap_ptr
                         tensor_dim_args,     // actual tensor dims
                        [llvm_i64_ty]),       // cache_hint
          [llvm_i1_ty],                       // Flag for cache_hint
          [IntrConvergent, ReadOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>,
           NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>]>;
  }
}

// Intrinsics for Prefetch and Prefetchu
let IntrProperties = [IntrArgMemOnly, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>] in {
  foreach level = ["L1", "L2"] in {
    def int_nvvm_prefetch_ # level : Intrinsic<[], [llvm_ptr_ty]>;
    def int_nvvm_prefetch_global_ # level : Intrinsic<[], [llvm_global_ptr_ty]>;
    def int_nvvm_prefetch_local_ # level : Intrinsic<[], [llvm_local_ptr_ty]>;
  }

  foreach eviction_priority = ["evict_normal", "evict_last"] in
    def int_nvvm_prefetch_global_L2_ # eviction_priority : Intrinsic<[], [llvm_global_ptr_ty]>;

  def int_nvvm_prefetchu_L1 : Intrinsic<[], [llvm_ptr_ty]>;
}

// applypriority
let IntrProperties = [IntrArgMemOnly, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
                      ImmArg<ArgIndex<1>>] in {
  def int_nvvm_applypriority_global_L2_evict_normal
    : DefaultAttrsIntrinsic<[], [llvm_global_ptr_ty, llvm_i64_ty]>;

  def int_nvvm_applypriority_L2_evict_normal
    : DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_i64_ty]>;
}

// discard
let IntrProperties = [NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<1>>, IntrHasSideEffects] in {
  def int_nvvm_discard_global_L2 : DefaultAttrsIntrinsic<[], [llvm_global_ptr_ty, llvm_i64_ty]>;
  def int_nvvm_discard_L2 : DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_i64_ty]>;
}

// Intrinsics for Bulk Copy using TMA (non-tensor)
// From Global to Shared Cluster
def int_nvvm_cp_async_bulk_global_to_shared_cluster
  : DefaultAttrsIntrinsicFlags<[],
      [llvm_shared_cluster_ptr_ty, // dst_shared_cluster_ptr
       llvm_shared_ptr_ty,         // mbarrier_ptr
       llvm_global_ptr_ty,         // src_gmem_ptr
       llvm_i32_ty,                // copy_size
       llvm_i16_ty,                // cta_mask
       llvm_i64_ty],               // cache_hint
      [llvm_i1_ty,                 // Flag for cta_mask
       llvm_i1_ty],                // Flag for cache_hint
      [IntrConvergent, IntrArgMemOnly,
       WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<2>>,
       NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>, NoCapture<ArgIndex<2>>]>;

// From Shared CTA to Shared Cluster
def int_nvvm_cp_async_bulk_shared_cta_to_cluster
  : DefaultAttrsIntrinsic<[],
      [llvm_shared_cluster_ptr_ty, // dst_shared_cluster_ptr
       llvm_shared_ptr_ty,         // mbarrier_ptr
       llvm_shared_ptr_ty,         // src_smem_ptr
       llvm_i32_ty],               // copy_size
      [IntrConvergent, IntrArgMemOnly,
       WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<2>>,
       NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>,
       NoCapture<ArgIndex<2>>]>;

// From Shared CTA to Global memory
def int_nvvm_cp_async_bulk_shared_cta_to_global
  : DefaultAttrsIntrinsicFlags<[],
      [llvm_global_ptr_ty, // dst_gmem_ptr
       llvm_shared_ptr_ty, // src_smem_ptr
       llvm_i32_ty,        // copy_size
       llvm_i64_ty],       // cache_hint
      [llvm_i1_ty],        // Flag for cache_hint
      [IntrConvergent, IntrArgMemOnly,
       WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>,
       NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>]>;

// From Shared CTA to Global memory with bytemask
def int_nvvm_cp_async_bulk_shared_cta_to_global_bytemask
  : DefaultAttrsIntrinsic<[],
      [llvm_global_ptr_ty, // dst_gmem_ptr
       llvm_shared_ptr_ty, // src_smem_ptr
       llvm_i32_ty,        // copy_size
       llvm_i64_ty,        // cache_hint
       llvm_i1_ty,         // Flag for cache_hint
       llvm_i16_ty],       // byte_mask
      [IntrConvergent, IntrArgMemOnly,
       WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>,
       ImmArg<ArgIndex<4>>]>;

// Intrinsics for Bulk Copy Prefetch L2
def int_nvvm_cp_async_bulk_prefetch_L2
  : DefaultAttrsIntrinsicFlags<[],
      [llvm_global_ptr_ty, // src_gmem_ptr
       llvm_i32_ty,        // copy_size
       llvm_i64_ty],       // cache_hint
      [llvm_i1_ty],        // Flag for cache_hint
      [IntrConvergent, IntrArgMemOnly,
       NoCapture<ArgIndex<0>>, ReadOnly<ArgIndex<0>>]>;

def int_nvvm_griddepcontrol_launch_dependents : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
def int_nvvm_griddepcontrol_wait : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

//
// Tcgen05 family of Intrinsics
//

// Tcgen05 alloc/dealloc related intrinsics

foreach cta_group = ["cg1", "cg2"] in {
  def int_nvvm_tcgen05_alloc_ # cta_group : Intrinsic<[],
    [llvm_ptr_ty,        // dst_ptr
     llvm_i32_ty] ,      // num_columns
    [IntrConvergent, IntrInaccessibleMemOrArgMemOnly,
     WriteOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_alloc_shared_ # cta_group : Intrinsic<[],
    [llvm_shared_ptr_ty, // dst_ptr
     llvm_i32_ty],       // num_columns
    [IntrConvergent, IntrInaccessibleMemOrArgMemOnly,
     WriteOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_dealloc_ # cta_group : Intrinsic<[],
    [llvm_tmem_ptr_ty,   // tmem_addr
     llvm_i32_ty],       // num_columns
    [IntrConvergent, IntrArgMemOnly,
     NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_relinq_alloc_permit_ # cta_group : Intrinsic<[], [],
    [IntrConvergent, IntrInaccessibleMemOnly]>;

  def int_nvvm_tcgen05_commit_ # cta_group : Intrinsic<[],
    [llvm_ptr_ty],        // mbar_ptr
    [IntrConvergent, IntrInaccessibleMemOrArgMemOnly,
     NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_commit_shared_ # cta_group : Intrinsic<[],
    [llvm_shared_ptr_ty], // mbar_ptr
    [IntrConvergent, IntrInaccessibleMemOrArgMemOnly,
     NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_commit_mc_ # cta_group : Intrinsic<[],
    [llvm_ptr_ty, llvm_i16_ty], // mbar_ptr, cta_mask
    [IntrConvergent, IntrInaccessibleMemOrArgMemOnly,
     NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_commit_mc_shared_ # cta_group : Intrinsic<[],
    [llvm_shared_ptr_ty, llvm_i16_ty], // mbar_ptr, cta_mask
    [IntrConvergent, IntrInaccessibleMemOrArgMemOnly,
     NoCapture<ArgIndex<0>>]>;

  def int_nvvm_tcgen05_shift_down_ # cta_group : Intrinsic<[],
    [llvm_tmem_ptr_ty],   // tmem_addr
    [IntrConvergent, IntrArgMemOnly,
     NoCapture<ArgIndex<0>>]>;
}

// Tcgen05 wait_ld/st intrinsics
def int_nvvm_tcgen05_wait_ld : Intrinsic<[], [],
  [IntrConvergent, IntrInaccessibleMemOnly]>;
def int_nvvm_tcgen05_wait_st : Intrinsic<[], [],
  [IntrConvergent, IntrInaccessibleMemOnly]>;

// Tcgen05 Fence intrinsics
def int_nvvm_tcgen05_fence_before_thread_sync : Intrinsic<[], [],
  [IntrNoMem, IntrHasSideEffects]>;
def int_nvvm_tcgen05_fence_after_thread_sync : Intrinsic<[], [],
  [IntrNoMem, IntrHasSideEffects]>;

// Tcgen05 cp intrinsics
foreach cta_group = ["cg1", "cg2"] in {
  foreach src_fmt = ["", "b6x16_p32", "b4x16_p64"] in {
    foreach shape = ["128x256b", "4x256b", "128x128b",
                     "64x128b_warpx2_02_13",
                     "64x128b_warpx2_01_23",
                     "32x128b_warpx4"] in {
      defvar intr_suffix = StrJoin<"_", [shape, src_fmt, cta_group]>.ret;
      defvar name_suffix = StrJoin<".", [shape, src_fmt, cta_group]>.ret;

      def int_nvvm_tcgen05_cp_ # intr_suffix : Intrinsic<[],
        [llvm_tmem_ptr_ty,   // tmem_addr
         llvm_i64_ty],       // smem descriptor
        [IntrConvergent, IntrInaccessibleMemOrArgMemOnly, NoCapture<ArgIndex<0>>],
        "llvm.nvvm.tcgen05.cp." # name_suffix>;
    }
  }
}

// Tcgen05 ld intrinsics
class NVVM_TCGEN05_LD<string Shape, int Num> :
        Intrinsic<[NVVM_TCGEN05_LDST_ACCESS_SIZE<Shape, Num>.type],
                  !listconcat([llvm_tmem_ptr_ty],
                              !if(!eq(Shape, "16x32bx2"), [llvm_i64_ty], []),
                              [llvm_i1_ty]),
                  !listconcat([IntrConvergent, IntrArgMemOnly, NoCapture<ArgIndex<0>>],
                              !if(!eq(Shape, "16x32bx2"),
                                [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>],
                                [ImmArg<ArgIndex<1>>]))>;

// Tcgen05 st intrinsics
class NVVM_TCGEN05_ST<string Shape, int Num> :
        Intrinsic<[],
                  !listconcat([llvm_tmem_ptr_ty],
                              !if(!eq(Shape, "16x32bx2"), [llvm_i64_ty], []),
                              [NVVM_TCGEN05_LDST_ACCESS_SIZE<Shape, Num>.type],
                              [llvm_i1_ty]),
                  !listconcat([IntrConvergent, IntrArgMemOnly, NoCapture<ArgIndex<0>>],
                              !if(!eq(Shape, "16x32bx2"),
                                [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>],
                                [ImmArg<ArgIndex<2>>]))>;

foreach shape = ["16x64b", "16x128b", "16x256b", "32x32b", "16x32bx2"] in {
  foreach num = 0...8 in {
    if NVVM_TCGEN05_LDST_ACCESS_SIZE<shape, num>.valid then {
      def int_nvvm_tcgen05_ld_ # shape # _x # !shl(1, num) :
            NVVM_TCGEN05_LD<shape, num>;
      def int_nvvm_tcgen05_st_ # shape # _x # !shl(1, num) :
            NVVM_TCGEN05_ST<shape, num>;
    }
  }
}

//
// Bulk store intrinsics
//
let IntrProperties = [IntrArgMemOnly, IntrWriteMem, WriteOnly<ArgIndex<0>>,
                      NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<2>>] in {
  def int_nvvm_st_bulk :
      DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_i64_ty, llvm_i64_ty]>;

  def int_nvvm_st_bulk_shared_cta :
      DefaultAttrsIntrinsic<[], [llvm_shared_ptr_ty, llvm_i64_ty, llvm_i64_ty]>;
}

//
// clusterlaunchcontorl Intrinsics
//

// clusterlaunchcontrol.try_cancel

def int_nvvm_clusterlaunchcontrol_try_cancel_async_shared
    : DefaultAttrsIntrinsic<[], [llvm_shared_ptr_ty, llvm_shared_ptr_ty],
                [IntrHasSideEffects, IntrArgMemOnly],
                "llvm.nvvm.clusterlaunchcontrol.try_cancel.async.shared">;

def int_nvvm_clusterlaunchcontrol_try_cancel_async_multicast_shared
    : DefaultAttrsIntrinsic<[], [llvm_shared_ptr_ty, llvm_shared_ptr_ty],
                [IntrHasSideEffects, IntrArgMemOnly],
                "llvm.nvvm.clusterlaunchcontrol.try_cancel.async.multicast.shared">;

// clusterlaunchcontrol.query_cancel.is_canceled

def int_nvvm_clusterlaunchcontrol_query_cancel_is_canceled
    : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i128_ty], [IntrNoMem, IntrSpeculatable],
                            "llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled">;

foreach dim = ["x", "y", "z"] in {
def int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_ # dim
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i128_ty], [IntrNoMem, IntrSpeculatable],
                            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid." # dim>;
}

} // let TargetPrefix = "nvvm"
