//===- llvm/Analysis/VectorUtils.h - Vector utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some vectorizer utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VECTORUTILS_H
#define LLVM_ANALYSIS_VECTORUTILS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace llvm {
class TargetLibraryInfo;

/// Describes the type of Parameters
enum class VFParamKind {
  Vector,            // No semantic information.
  OMP_Linear,        // declare simd linear(i)
  OMP_LinearRef,     // declare simd linear(ref(i))
  OMP_LinearVal,     // declare simd linear(val(i))
  OMP_LinearUVal,    // declare simd linear(uval(i))
  OMP_LinearPos,     // declare simd linear(i:c) uniform(c)
  OMP_LinearValPos,  // declare simd linear(val(i:c)) uniform(c)
  OMP_LinearRefPos,  // declare simd linear(ref(i:c)) uniform(c)
  OMP_LinearUValPos, // declare simd linear(uval(i:c)) uniform(c)
  OMP_Uniform,       // declare simd uniform(i)
  GlobalPredicate,   // Global logical predicate that acts on all lanes
                     // of the input and output mask concurrently. For
                     // example, it is implied by the `M` token in the
                     // Vector Function ABI mangled name.
  Unknown
};

/// Describes the type of Instruction Set Architecture
enum class VFISAKind {
  AdvancedSIMD, // AArch64 Advanced SIMD (NEON)
  SVE,          // AArch64 Scalable Vector Extension
  SSE,          // x86 SSE
  AVX,          // x86 AVX
  AVX2,         // x86 AVX2
  AVX512,       // x86 AVX512
  LLVM,         // LLVM internal ISA for functions that are not
  // attached to an existing ABI via name mangling.
  Unknown // Unknown ISA
};

/// Encapsulates information needed to describe a parameter.
///
/// The description of the parameter is not linked directly to
/// OpenMP or any other vector function description. This structure
/// is extendible to handle other paradigms that describe vector
/// functions and their parameters.
struct VFParameter {
  unsigned ParamPos;         // Parameter Position in Scalar Function.
  VFParamKind ParamKind;     // Kind of Parameter.
  int LinearStepOrPos = 0;   // Step or Position of the Parameter.
  Align Alignment = Align(); // Optional alignment in bytes, defaulted to 1.

  // Comparison operator.
  bool operator==(const VFParameter &Other) const {
    return std::tie(ParamPos, ParamKind, LinearStepOrPos, Alignment) ==
           std::tie(Other.ParamPos, Other.ParamKind, Other.LinearStepOrPos,
                    Other.Alignment);
  }
};

/// Contains the information about the kind of vectorization
/// available.
///
/// This object in independent on the paradigm used to
/// represent vector functions. in particular, it is not attached to
/// any target-specific ABI.
struct VFShape {
  ElementCount VF;                        // Vectorization factor.
  SmallVector<VFParameter, 8> Parameters; // List of parameter information.
  // Comparison operator.
  bool operator==(const VFShape &Other) const {
    return std::tie(VF, Parameters) == std::tie(Other.VF, Other.Parameters);
  }

  /// Update the parameter in position P.ParamPos to P.
  void updateParam(VFParameter P) {
    assert(P.ParamPos < Parameters.size() && "Invalid parameter position.");
    Parameters[P.ParamPos] = P;
    assert(hasValidParameterList() && "Invalid parameter list");
  }

  // Retrieve the VFShape that can be used to map a (scalar) function to itself,
  // with VF = 1.
  static VFShape getScalarShape(const CallInst &CI) {
    return VFShape::get(CI, ElementCount::getFixed(1),
                        /*HasGlobalPredicate*/ false);
  }

  // Retrieve the basic vectorization shape of the function, where all
  // parameters are mapped to VFParamKind::Vector with \p EC
  // lanes. Specifies whether the function has a Global Predicate
  // argument via \p HasGlobalPred.
  static VFShape get(const CallInst &CI, ElementCount EC, bool HasGlobalPred) {
    SmallVector<VFParameter, 8> Parameters;
    for (unsigned I = 0; I < CI.arg_size(); ++I)
      Parameters.push_back(VFParameter({I, VFParamKind::Vector}));
    if (HasGlobalPred)
      Parameters.push_back(
          VFParameter({CI.arg_size(), VFParamKind::GlobalPredicate}));

    return {EC, Parameters};
  }
  /// Validation check on the Parameters in the VFShape.
  bool hasValidParameterList() const;
};

/// Holds the VFShape for a specific scalar to vector function mapping.
struct VFInfo {
  VFShape Shape;          /// Classification of the vector function.
  std::string ScalarName; /// Scalar Function Name.
  std::string VectorName; /// Vector Function Name associated to this VFInfo.
  VFISAKind ISA;          /// Instruction Set Architecture.
};

namespace VFABI {
/// LLVM Internal VFABI ISA token for vector functions.
static constexpr char const *_LLVM_ = "_LLVM_";
/// Prefix for internal name redirection for vector function that
/// tells the compiler to scalarize the call using the scalar name
/// of the function. For example, a mangled name like
/// `_ZGV_LLVM_N2v_foo(_LLVM_Scalarize_foo)` would tell the
/// vectorizer to vectorize the scalar call `foo`, and to scalarize
/// it once vectorization is done.
static constexpr char const *_LLVM_Scalarize_ = "_LLVM_Scalarize_";

/// Function to construct a VFInfo out of a mangled names in the
/// following format:
///
/// <VFABI_name>{(<redirection>)}
///
/// where <VFABI_name> is the name of the vector function, mangled according
/// to the rules described in the Vector Function ABI of the target vector
/// extension (or <isa> from now on). The <VFABI_name> is in the following
/// format:
///
/// _ZGV<isa><mask><vlen><parameters>_<scalarname>[(<redirection>)]
///
/// This methods support demangling rules for the following <isa>:
///
/// * AArch64: https://developer.arm.com/docs/101129/latest
///
/// * x86 (libmvec): https://sourceware.org/glibc/wiki/libmvec and
///  https://sourceware.org/glibc/wiki/libmvec?action=AttachFile&do=view&target=VectorABI.txt
///
/// \param MangledName -> input string in the format
/// _ZGV<isa><mask><vlen><parameters>_<scalarname>[(<redirection>)].
/// \param M -> Module used to retrieve informations about the vector
/// function that are not possible to retrieve from the mangled
/// name. At the moment, this parameter is needed only to retrieve the
/// Vectorization Factor of scalable vector functions from their
/// respective IR declarations.
Optional<VFInfo> tryDemangleForVFABI(StringRef MangledName, const Module &M);

/// This routine mangles the given VectorName according to the LangRef
/// specification for vector-function-abi-variant attribute and is specific to
/// the TLI mappings. It is the responsibility of the caller to make sure that
/// this is only used if all parameters in the vector function are vector type.
/// This returned string holds scalar-to-vector mapping:
///    _ZGV<isa><mask><vlen><vparams>_<scalarname>(<vectorname>)
///
/// where:
///
/// <isa> = "_LLVM_"
/// <mask> = "N". Note: TLI does not support masked interfaces.
/// <vlen> = Number of concurrent lanes, stored in the `VectorizationFactor`
///          field of the `VecDesc` struct. If the number of lanes is scalable
///          then 'x' is printed instead.
/// <vparams> = "v", as many as are the numArgs.
/// <scalarname> = the name of the scalar function.
/// <vectorname> = the name of the vector function.
std::string mangleTLIVectorName(StringRef VectorName, StringRef ScalarName,
                                unsigned numArgs, ElementCount VF);

/// Retrieve the `VFParamKind` from a string token.
VFParamKind getVFParamKindFromString(const StringRef Token);

// Name of the attribute where the variant mappings are stored.
static constexpr char const *MappingsAttrName = "vector-function-abi-variant";

/// Populates a set of strings representing the Vector Function ABI variants
/// associated to the CallInst CI. If the CI does not contain the
/// vector-function-abi-variant attribute, we return without populating
/// VariantMappings, i.e. callers of getVectorVariantNames need not check for
/// the presence of the attribute (see InjectTLIMappings).
void getVectorVariantNames(const CallInst &CI,
                           SmallVectorImpl<std::string> &VariantMappings);
} // end namespace VFABI

/// The Vector Function Database.
///
/// Helper class used to find the vector functions associated to a
/// scalar CallInst.
class VFDatabase {
  /// The Module of the CallInst CI.
  const Module *M;
  /// The CallInst instance being queried for scalar to vector mappings.
  const CallInst &CI;
  /// List of vector functions descriptors associated to the call
  /// instruction.
  const SmallVector<VFInfo, 8> ScalarToVectorMappings;

  /// Retrieve the scalar-to-vector mappings associated to the rule of
  /// a vector Function ABI.
  static void getVFABIMappings(const CallInst &CI,
                               SmallVectorImpl<VFInfo> &Mappings) {
    if (!CI.getCalledFunction())
      return;

    const StringRef ScalarName = CI.getCalledFunction()->getName();

    SmallVector<std::string, 8> ListOfStrings;
    // The check for the vector-function-abi-variant attribute is done when
    // retrieving the vector variant names here.
    VFABI::getVectorVariantNames(CI, ListOfStrings);
    if (ListOfStrings.empty())
      return;
    for (const auto &MangledName : ListOfStrings) {
      const Optional<VFInfo> Shape =
          VFABI::tryDemangleForVFABI(MangledName, *(CI.getModule()));
      // A match is found via scalar and vector names, and also by
      // ensuring that the variant described in the attribute has a
      // corresponding definition or declaration of the vector
      // function in the Module M.
      if (Shape && (Shape.value().ScalarName == ScalarName)) {
        assert(CI.getModule()->getFunction(Shape.value().VectorName) &&
               "Vector function is missing.");
        Mappings.push_back(Shape.value());
      }
    }
  }

public:
  /// Retrieve all the VFInfo instances associated to the CallInst CI.
  static SmallVector<VFInfo, 8> getMappings(const CallInst &CI) {
    SmallVector<VFInfo, 8> Ret;

    // Get mappings from the Vector Function ABI variants.
    getVFABIMappings(CI, Ret);

    // Other non-VFABI variants should be retrieved here.

    return Ret;
  }

  /// Constructor, requires a CallInst instance.
  VFDatabase(CallInst &CI)
      : M(CI.getModule()), CI(CI),
        ScalarToVectorMappings(VFDatabase::getMappings(CI)) {}
  /// \defgroup VFDatabase query interface.
  ///
  /// @{
  /// Retrieve the Function with VFShape \p Shape.
  Function *getVectorizedFunction(const VFShape &Shape) const {
    if (Shape == VFShape::getScalarShape(CI))
      return CI.getCalledFunction();

    for (const auto &Info : ScalarToVectorMappings)
      if (Info.Shape == Shape)
        return M->getFunction(Info.VectorName);

    return nullptr;
  }
  /// @}
};

template <typename T> class ArrayRef;
class DemandedBits;
class GetElementPtrInst;
template <typename InstTy> class InterleaveGroup;
class IRBuilderBase;
class Loop;
class ScalarEvolution;
class TargetTransformInfo;
class Type;
class Value;

namespace Intrinsic {
typedef unsigned ID;
}

/// A helper function for converting Scalar types to vector types. If
/// the incoming type is void, we return void. If the EC represents a
/// scalar, we return the scalar type.
inline Type *ToVectorTy(Type *Scalar, ElementCount EC) {
  if (Scalar->isVoidTy() || Scalar->isMetadataTy() || EC.isScalar())
    return Scalar;
  return VectorType::get(Scalar, EC);
}

inline Type *ToVectorTy(Type *Scalar, unsigned VF) {
  return ToVectorTy(Scalar, ElementCount::getFixed(VF));
}

/// Identify if the intrinsic is trivially vectorizable.
/// This method returns true if the intrinsic's argument types are all scalars
/// for the scalar form of the intrinsic and all vectors (or scalars handled by
/// isVectorIntrinsicWithScalarOpAtArg) for the vector form of the intrinsic.
bool isTriviallyVectorizable(Intrinsic::ID ID);

/// Identifies if the vector form of the intrinsic has a scalar operand.
bool isVectorIntrinsicWithScalarOpAtArg(Intrinsic::ID ID,
                                        unsigned ScalarOpdIdx);

/// Identifies if the vector form of the intrinsic has a operand that has
/// an overloaded type.
bool isVectorIntrinsicWithOverloadTypeAtArg(Intrinsic::ID ID, unsigned OpdIdx);

/// Returns intrinsic ID for call.
/// For the input call instruction it finds mapping intrinsic and returns
/// its intrinsic ID, in case it does not found it return not_intrinsic.
Intrinsic::ID getVectorIntrinsicIDForCall(const CallInst *CI,
                                          const TargetLibraryInfo *TLI);

/// Find the operand of the GEP that should be checked for consecutive
/// stores. This ignores trailing indices that have no effect on the final
/// pointer.
unsigned getGEPInductionOperand(const GetElementPtrInst *Gep);

/// If the argument is a GEP, then returns the operand identified by
/// getGEPInductionOperand. However, if there is some other non-loop-invariant
/// operand, it returns that instead.
Value *stripGetElementPtr(Value *Ptr, ScalarEvolution *SE, Loop *Lp);

/// If a value has only one user that is a CastInst, return it.
Value *getUniqueCastUse(Value *Ptr, Loop *Lp, Type *Ty);

/// Get the stride of a pointer access in a loop. Looks for symbolic
/// strides "a[i*stride]". Returns the symbolic stride, or null otherwise.
Value *getStrideFromPointer(Value *Ptr, ScalarEvolution *SE, Loop *Lp);

/// Given a vector and an element number, see if the scalar value is
/// already around as a register, for example if it were inserted then extracted
/// from the vector.
Value *findScalarElement(Value *V, unsigned EltNo);

/// If all non-negative \p Mask elements are the same value, return that value.
/// If all elements are negative (undefined) or \p Mask contains different
/// non-negative values, return -1.
int getSplatIndex(ArrayRef<int> Mask);

/// Get splat value if the input is a splat vector or return nullptr.
/// The value may be extracted from a splat constants vector or from
/// a sequence of instructions that broadcast a single value into a vector.
Value *getSplatValue(const Value *V);

/// Return true if each element of the vector value \p V is poisoned or equal to
/// every other non-poisoned element. If an index element is specified, either
/// every element of the vector is poisoned or the element at that index is not
/// poisoned and equal to every other non-poisoned element.
/// This may be more powerful than the related getSplatValue() because it is
/// not limited by finding a scalar source value to a splatted vector.
bool isSplatValue(const Value *V, int Index = -1, unsigned Depth = 0);

/// Replace each shuffle mask index with the scaled sequential indices for an
/// equivalent mask of narrowed elements. Mask elements that are less than 0
/// (sentinel values) are repeated in the output mask.
///
/// Example with Scale = 4:
///   <4 x i32> <3, 2, 0, -1> -->
///   <16 x i8> <12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, -1, -1, -1, -1>
///
/// This is the reverse process of widening shuffle mask elements, but it always
/// succeeds because the indexes can always be multiplied (scaled up) to map to
/// narrower vector elements.
void narrowShuffleMaskElts(int Scale, ArrayRef<int> Mask,
                           SmallVectorImpl<int> &ScaledMask);

/// Try to transform a shuffle mask by replacing elements with the scaled index
/// for an equivalent mask of widened elements. If all mask elements that would
/// map to a wider element of the new mask are the same negative number
/// (sentinel value), that element of the new mask is the same value. If any
/// element in a given slice is negative and some other element in that slice is
/// not the same value, return false (partial matches with sentinel values are
/// not allowed).
///
/// Example with Scale = 4:
///   <16 x i8> <12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, -1, -1, -1, -1> -->
///   <4 x i32> <3, 2, 0, -1>
///
/// This is the reverse process of narrowing shuffle mask elements if it
/// succeeds. This transform is not always possible because indexes may not
/// divide evenly (scale down) to map to wider vector elements.
bool widenShuffleMaskElts(int Scale, ArrayRef<int> Mask,
                          SmallVectorImpl<int> &ScaledMask);

/// Splits and processes shuffle mask depending on the number of input and
/// output registers. The function does 2 main things: 1) splits the
/// source/destination vectors into real registers; 2) do the mask analysis to
/// identify which real registers are permuted. Then the function processes
/// resulting registers mask using provided action items. If no input register
/// is defined, \p NoInputAction action is used. If only 1 input register is
/// used, \p SingleInputAction is used, otherwise \p ManyInputsAction is used to
/// process > 2 input registers and masks.
/// \param Mask Original shuffle mask.
/// \param NumOfSrcRegs Number of source registers.
/// \param NumOfDestRegs Number of destination registers.
/// \param NumOfUsedRegs Number of actually used destination registers.
void processShuffleMasks(
    ArrayRef<int> Mask, unsigned NumOfSrcRegs, unsigned NumOfDestRegs,
    unsigned NumOfUsedRegs, function_ref<void()> NoInputAction,
    function_ref<void(ArrayRef<int>, unsigned, unsigned)> SingleInputAction,
    function_ref<void(ArrayRef<int>, unsigned, unsigned)> ManyInputsAction);

/// Compute a map of integer instructions to their minimum legal type
/// size.
///
/// C semantics force sub-int-sized values (e.g. i8, i16) to be promoted to int
/// type (e.g. i32) whenever arithmetic is performed on them.
///
/// For targets with native i8 or i16 operations, usually InstCombine can shrink
/// the arithmetic type down again. However InstCombine refuses to create
/// illegal types, so for targets without i8 or i16 registers, the lengthening
/// and shrinking remains.
///
/// Most SIMD ISAs (e.g. NEON) however support vectors of i8 or i16 even when
/// their scalar equivalents do not, so during vectorization it is important to
/// remove these lengthens and truncates when deciding the profitability of
/// vectorization.
///
/// This function analyzes the given range of instructions and determines the
/// minimum type size each can be converted to. It attempts to remove or
/// minimize type size changes across each def-use chain, so for example in the
/// following code:
///
///   %1 = load i8, i8*
///   %2 = add i8 %1, 2
///   %3 = load i16, i16*
///   %4 = zext i8 %2 to i32
///   %5 = zext i16 %3 to i32
///   %6 = add i32 %4, %5
///   %7 = trunc i32 %6 to i16
///
/// Instruction %6 must be done at least in i16, so computeMinimumValueSizes
/// will return: {%1: 16, %2: 16, %3: 16, %4: 16, %5: 16, %6: 16, %7: 16}.
///
/// If the optional TargetTransformInfo is provided, this function tries harder
/// to do less work by only looking at illegal types.
MapVector<Instruction*, uint64_t>
computeMinimumValueSizes(ArrayRef<BasicBlock*> Blocks,
                         DemandedBits &DB,
                         const TargetTransformInfo *TTI=nullptr);

/// Compute the union of two access-group lists.
///
/// If the list contains just one access group, it is returned directly. If the
/// list is empty, returns nullptr.
MDNode *uniteAccessGroups(MDNode *AccGroups1, MDNode *AccGroups2);

/// Compute the access-group list of access groups that @p Inst1 and @p Inst2
/// are both in. If either instruction does not access memory at all, it is
/// considered to be in every list.
///
/// If the list contains just one access group, it is returned directly. If the
/// list is empty, returns nullptr.
MDNode *intersectAccessGroups(const Instruction *Inst1,
                              const Instruction *Inst2);

/// Specifically, let Kinds = [MD_tbaa, MD_alias_scope, MD_noalias, MD_fpmath,
/// MD_nontemporal, MD_access_group].
/// For K in Kinds, we get the MDNode for K from each of the
/// elements of VL, compute their "intersection" (i.e., the most generic
/// metadata value that covers all of the individual values), and set I's
/// metadata for M equal to the intersection value.
///
/// This function always sets a (possibly null) value for each K in Kinds.
Instruction *propagateMetadata(Instruction *I, ArrayRef<Value *> VL);

/// Create a mask that filters the members of an interleave group where there
/// are gaps.
///
/// For example, the mask for \p Group with interleave-factor 3
/// and \p VF 4, that has only its first member present is:
///
///   <1,0,0,1,0,0,1,0,0,1,0,0>
///
/// Note: The result is a mask of 0's and 1's, as opposed to the other
/// create[*]Mask() utilities which create a shuffle mask (mask that
/// consists of indices).
Constant *createBitMaskForGaps(IRBuilderBase &Builder, unsigned VF,
                               const InterleaveGroup<Instruction> &Group);

/// Create a mask with replicated elements.
///
/// This function creates a shuffle mask for replicating each of the \p VF
/// elements in a vector \p ReplicationFactor times. It can be used to
/// transform a mask of \p VF elements into a mask of
/// \p VF * \p ReplicationFactor elements used by a predicated
/// interleaved-group of loads/stores whose Interleaved-factor ==
/// \p ReplicationFactor.
///
/// For example, the mask for \p ReplicationFactor=3 and \p VF=4 is:
///
///   <0,0,0,1,1,1,2,2,2,3,3,3>
llvm::SmallVector<int, 16> createReplicatedMask(unsigned ReplicationFactor,
                                                unsigned VF);

/// Create an interleave shuffle mask.
///
/// This function creates a shuffle mask for interleaving \p NumVecs vectors of
/// vectorization factor \p VF into a single wide vector. The mask is of the
/// form:
///
///   <0, VF, VF * 2, ..., VF * (NumVecs - 1), 1, VF + 1, VF * 2 + 1, ...>
///
/// For example, the mask for VF = 4 and NumVecs = 2 is:
///
///   <0, 4, 1, 5, 2, 6, 3, 7>.
llvm::SmallVector<int, 16> createInterleaveMask(unsigned VF, unsigned NumVecs);

/// Create a stride shuffle mask.
///
/// This function creates a shuffle mask whose elements begin at \p Start and
/// are incremented by \p Stride. The mask can be used to deinterleave an
/// interleaved vector into separate vectors of vectorization factor \p VF. The
/// mask is of the form:
///
///   <Start, Start + Stride, ..., Start + Stride * (VF - 1)>
///
/// For example, the mask for Start = 0, Stride = 2, and VF = 4 is:
///
///   <0, 2, 4, 6>
llvm::SmallVector<int, 16> createStrideMask(unsigned Start, unsigned Stride,
                                            unsigned VF);

/// Create a sequential shuffle mask.
///
/// This function creates shuffle mask whose elements are sequential and begin
/// at \p Start.  The mask contains \p NumInts integers and is padded with \p
/// NumUndefs undef values. The mask is of the form:
///
///   <Start, Start + 1, ... Start + NumInts - 1, undef_1, ... undef_NumUndefs>
///
/// For example, the mask for Start = 0, NumInsts = 4, and NumUndefs = 4 is:
///
///   <0, 1, 2, 3, undef, undef, undef, undef>
llvm::SmallVector<int, 16>
createSequentialMask(unsigned Start, unsigned NumInts, unsigned NumUndefs);

/// Given a shuffle mask for a binary shuffle, create the equivalent shuffle
/// mask assuming both operands are identical. This assumes that the unary
/// shuffle will use elements from operand 0 (operand 1 will be unused).
llvm::SmallVector<int, 16> createUnaryMask(ArrayRef<int> Mask,
                                           unsigned NumElts);

/// Concatenate a list of vectors.
///
/// This function generates code that concatenate the vectors in \p Vecs into a
/// single large vector. The number of vectors should be greater than one, and
/// their element types should be the same. The number of elements in the
/// vectors should also be the same; however, if the last vector has fewer
/// elements, it will be padded with undefs.
Value *concatenateVectors(IRBuilderBase &Builder, ArrayRef<Value *> Vecs);

/// Given a mask vector of i1, Return true if all of the elements of this
/// predicate mask are known to be false or undef.  That is, return true if all
/// lanes can be assumed inactive.
bool maskIsAllZeroOrUndef(Value *Mask);

/// Given a mask vector of i1, Return true if all of the elements of this
/// predicate mask are known to be true or undef.  That is, return true if all
/// lanes can be assumed active.
bool maskIsAllOneOrUndef(Value *Mask);

/// Given a mask vector of the form <Y x i1>, return an APInt (of bitwidth Y)
/// for each lane which may be active.
APInt possiblyDemandedEltsInMask(Value *Mask);

/// The group of interleaved loads/stores sharing the same stride and
/// close to each other.
///
/// Each member in this group has an index starting from 0, and the largest
/// index should be less than interleaved factor, which is equal to the absolute
/// value of the access's stride.
///
/// E.g. An interleaved load group of factor 4:
///        for (unsigned i = 0; i < 1024; i+=4) {
///          a = A[i];                           // Member of index 0
///          b = A[i+1];                         // Member of index 1
///          d = A[i+3];                         // Member of index 3
///          ...
///        }
///
///      An interleaved store group of factor 4:
///        for (unsigned i = 0; i < 1024; i+=4) {
///          ...
///          A[i]   = a;                         // Member of index 0
///          A[i+1] = b;                         // Member of index 1
///          A[i+2] = c;                         // Member of index 2
///          A[i+3] = d;                         // Member of index 3
///        }
///
/// Note: the interleaved load group could have gaps (missing members), but
/// the interleaved store group doesn't allow gaps.
template <typename InstTy> class InterleaveGroup {
public:
  InterleaveGroup(uint32_t Factor, bool Reverse, Align Alignment)
      : Factor(Factor), Reverse(Reverse), Alignment(Alignment),
        InsertPos(nullptr) {}

  InterleaveGroup(InstTy *Instr, int32_t Stride, Align Alignment)
      : Alignment(Alignment), InsertPos(Instr) {
    Factor = std::abs(Stride);
    assert(Factor > 1 && "Invalid interleave factor");

    Reverse = Stride < 0;
    Members[0] = Instr;
  }

  bool isReverse() const { return Reverse; }
  uint32_t getFactor() const { return Factor; }
  Align getAlign() const { return Alignment; }
  uint32_t getNumMembers() const { return Members.size(); }

  /// Try to insert a new member \p Instr with index \p Index and
  /// alignment \p NewAlign. The index is related to the leader and it could be
  /// negative if it is the new leader.
  ///
  /// \returns false if the instruction doesn't belong to the group.
  bool insertMember(InstTy *Instr, int32_t Index, Align NewAlign) {
    // Make sure the key fits in an int32_t.
    Optional<int32_t> MaybeKey = checkedAdd(Index, SmallestKey);
    if (!MaybeKey)
      return false;
    int32_t Key = *MaybeKey;

    // Skip if the key is used for either the tombstone or empty special values.
    if (DenseMapInfo<int32_t>::getTombstoneKey() == Key ||
        DenseMapInfo<int32_t>::getEmptyKey() == Key)
      return false;

    // Skip if there is already a member with the same index.
    if (Members.find(Key) != Members.end())
      return false;

    if (Key > LargestKey) {
      // The largest index is always less than the interleave factor.
      if (Index >= static_cast<int32_t>(Factor))
        return false;

      LargestKey = Key;
    } else if (Key < SmallestKey) {

      // Make sure the largest index fits in an int32_t.
      Optional<int32_t> MaybeLargestIndex = checkedSub(LargestKey, Key);
      if (!MaybeLargestIndex)
        return false;

      // The largest index is always less than the interleave factor.
      if (*MaybeLargestIndex >= static_cast<int64_t>(Factor))
        return false;

      SmallestKey = Key;
    }

    // It's always safe to select the minimum alignment.
    Alignment = std::min(Alignment, NewAlign);
    Members[Key] = Instr;
    return true;
  }

  /// Get the member with the given index \p Index
  ///
  /// \returns nullptr if contains no such member.
  InstTy *getMember(uint32_t Index) const {
    int32_t Key = SmallestKey + Index;
    return Members.lookup(Key);
  }

  /// Get the index for the given member. Unlike the key in the member
  /// map, the index starts from 0.
  uint32_t getIndex(const InstTy *Instr) const {
    for (auto I : Members) {
      if (I.second == Instr)
        return I.first - SmallestKey;
    }

    llvm_unreachable("InterleaveGroup contains no such member");
  }

  InstTy *getInsertPos() const { return InsertPos; }
  void setInsertPos(InstTy *Inst) { InsertPos = Inst; }

  /// Add metadata (e.g. alias info) from the instructions in this group to \p
  /// NewInst.
  ///
  /// FIXME: this function currently does not add noalias metadata a'la
  /// addNewMedata.  To do that we need to compute the intersection of the
  /// noalias info from all members.
  void addMetadata(InstTy *NewInst) const;

  /// Returns true if this Group requires a scalar iteration to handle gaps.
  bool requiresScalarEpilogue() const {
    // If the last member of the Group exists, then a scalar epilog is not
    // needed for this group.
    if (getMember(getFactor() - 1))
      return false;

    // We have a group with gaps. It therefore can't be a reversed access,
    // because such groups get invalidated (TODO).
    assert(!isReverse() && "Group should have been invalidated");

    // This is a group of loads, with gaps, and without a last-member
    return true;
  }

private:
  uint32_t Factor; // Interleave Factor.
  bool Reverse;
  Align Alignment;
  DenseMap<int32_t, InstTy *> Members;
  int32_t SmallestKey = 0;
  int32_t LargestKey = 0;

  // To avoid breaking dependences, vectorized instructions of an interleave
  // group should be inserted at either the first load or the last store in
  // program order.
  //
  // E.g. %even = load i32             // Insert Position
  //      %add = add i32 %even         // Use of %even
  //      %odd = load i32
  //
  //      store i32 %even
  //      %odd = add i32               // Def of %odd
  //      store i32 %odd               // Insert Position
  InstTy *InsertPos;
};

/// Drive the analysis of interleaved memory accesses in the loop.
///
/// Use this class to analyze interleaved accesses only when we can vectorize
/// a loop. Otherwise it's meaningless to do analysis as the vectorization
/// on interleaved accesses is unsafe.
///
/// The analysis collects interleave groups and records the relationships
/// between the member and the group in a map.
class InterleavedAccessInfo {
public:
  InterleavedAccessInfo(PredicatedScalarEvolution &PSE, Loop *L,
                        DominatorTree *DT, LoopInfo *LI,
                        const LoopAccessInfo *LAI)
      : PSE(PSE), TheLoop(L), DT(DT), LI(LI), LAI(LAI) {}

  ~InterleavedAccessInfo() { invalidateGroups(); }

  /// Analyze the interleaved accesses and collect them in interleave
  /// groups. Substitute symbolic strides using \p Strides.
  /// Consider also predicated loads/stores in the analysis if
  /// \p EnableMaskedInterleavedGroup is true.
  void analyzeInterleaving(bool EnableMaskedInterleavedGroup);

  /// Invalidate groups, e.g., in case all blocks in loop will be predicated
  /// contrary to original assumption. Although we currently prevent group
  /// formation for predicated accesses, we may be able to relax this limitation
  /// in the future once we handle more complicated blocks. Returns true if any
  /// groups were invalidated.
  bool invalidateGroups() {
    if (InterleaveGroups.empty()) {
      assert(
          !RequiresScalarEpilogue &&
          "RequiresScalarEpilog should not be set without interleave groups");
      return false;
    }

    InterleaveGroupMap.clear();
    for (auto *Ptr : InterleaveGroups)
      delete Ptr;
    InterleaveGroups.clear();
    RequiresScalarEpilogue = false;
    return true;
  }

  /// Check if \p Instr belongs to any interleave group.
  bool isInterleaved(Instruction *Instr) const {
    return InterleaveGroupMap.find(Instr) != InterleaveGroupMap.end();
  }

  /// Get the interleave group that \p Instr belongs to.
  ///
  /// \returns nullptr if doesn't have such group.
  InterleaveGroup<Instruction> *
  getInterleaveGroup(const Instruction *Instr) const {
    return InterleaveGroupMap.lookup(Instr);
  }

  iterator_range<SmallPtrSetIterator<llvm::InterleaveGroup<Instruction> *>>
  getInterleaveGroups() {
    return make_range(InterleaveGroups.begin(), InterleaveGroups.end());
  }

  /// Returns true if an interleaved group that may access memory
  /// out-of-bounds requires a scalar epilogue iteration for correctness.
  bool requiresScalarEpilogue() const { return RequiresScalarEpilogue; }

  /// Invalidate groups that require a scalar epilogue (due to gaps). This can
  /// happen when optimizing for size forbids a scalar epilogue, and the gap
  /// cannot be filtered by masking the load/store.
  void invalidateGroupsRequiringScalarEpilogue();

  /// Returns true if we have any interleave groups.
  bool hasGroups() const { return !InterleaveGroups.empty(); }

private:
  /// A wrapper around ScalarEvolution, used to add runtime SCEV checks.
  /// Simplifies SCEV expressions in the context of existing SCEV assumptions.
  /// The interleaved access analysis can also add new predicates (for example
  /// by versioning strides of pointers).
  PredicatedScalarEvolution &PSE;

  Loop *TheLoop;
  DominatorTree *DT;
  LoopInfo *LI;
  const LoopAccessInfo *LAI;

  /// True if the loop may contain non-reversed interleaved groups with
  /// out-of-bounds accesses. We ensure we don't speculatively access memory
  /// out-of-bounds by executing at least one scalar epilogue iteration.
  bool RequiresScalarEpilogue = false;

  /// Holds the relationships between the members and the interleave group.
  DenseMap<Instruction *, InterleaveGroup<Instruction> *> InterleaveGroupMap;

  SmallPtrSet<InterleaveGroup<Instruction> *, 4> InterleaveGroups;

  /// Holds dependences among the memory accesses in the loop. It maps a source
  /// access to a set of dependent sink accesses.
  DenseMap<Instruction *, SmallPtrSet<Instruction *, 2>> Dependences;

  /// The descriptor for a strided memory access.
  struct StrideDescriptor {
    StrideDescriptor() = default;
    StrideDescriptor(int64_t Stride, const SCEV *Scev, uint64_t Size,
                     Align Alignment)
        : Stride(Stride), Scev(Scev), Size(Size), Alignment(Alignment) {}

    // The access's stride. It is negative for a reverse access.
    int64_t Stride = 0;

    // The scalar expression of this access.
    const SCEV *Scev = nullptr;

    // The size of the memory object.
    uint64_t Size = 0;

    // The alignment of this access.
    Align Alignment;
  };

  /// A type for holding instructions and their stride descriptors.
  using StrideEntry = std::pair<Instruction *, StrideDescriptor>;

  /// Create a new interleave group with the given instruction \p Instr,
  /// stride \p Stride and alignment \p Align.
  ///
  /// \returns the newly created interleave group.
  InterleaveGroup<Instruction> *
  createInterleaveGroup(Instruction *Instr, int Stride, Align Alignment) {
    assert(!InterleaveGroupMap.count(Instr) &&
           "Already in an interleaved access group");
    InterleaveGroupMap[Instr] =
        new InterleaveGroup<Instruction>(Instr, Stride, Alignment);
    InterleaveGroups.insert(InterleaveGroupMap[Instr]);
    return InterleaveGroupMap[Instr];
  }

  /// Release the group and remove all the relationships.
  void releaseGroup(InterleaveGroup<Instruction> *Group) {
    for (unsigned i = 0; i < Group->getFactor(); i++)
      if (Instruction *Member = Group->getMember(i))
        InterleaveGroupMap.erase(Member);

    InterleaveGroups.erase(Group);
    delete Group;
  }

  /// Collect all the accesses with a constant stride in program order.
  void collectConstStrideAccesses(
      MapVector<Instruction *, StrideDescriptor> &AccessStrideInfo,
      const ValueToValueMap &Strides);

  /// Returns true if \p Stride is allowed in an interleaved group.
  static bool isStrided(int Stride);

  /// Returns true if \p BB is a predicated block.
  bool isPredicated(BasicBlock *BB) const {
    return LoopAccessInfo::blockNeedsPredication(BB, TheLoop, DT);
  }

  /// Returns true if LoopAccessInfo can be used for dependence queries.
  bool areDependencesValid() const {
    return LAI && LAI->getDepChecker().getDependences();
  }

  /// Returns true if memory accesses \p A and \p B can be reordered, if
  /// necessary, when constructing interleaved groups.
  ///
  /// \p A must precede \p B in program order. We return false if reordering is
  /// not necessary or is prevented because \p A and \p B may be dependent.
  bool canReorderMemAccessesForInterleavedGroups(StrideEntry *A,
                                                 StrideEntry *B) const {
    // Code motion for interleaved accesses can potentially hoist strided loads
    // and sink strided stores. The code below checks the legality of the
    // following two conditions:
    //
    // 1. Potentially moving a strided load (B) before any store (A) that
    //    precedes B, or
    //
    // 2. Potentially moving a strided store (A) after any load or store (B)
    //    that A precedes.
    //
    // It's legal to reorder A and B if we know there isn't a dependence from A
    // to B. Note that this determination is conservative since some
    // dependences could potentially be reordered safely.

    // A is potentially the source of a dependence.
    auto *Src = A->first;
    auto SrcDes = A->second;

    // B is potentially the sink of a dependence.
    auto *Sink = B->first;
    auto SinkDes = B->second;

    // Code motion for interleaved accesses can't violate WAR dependences.
    // Thus, reordering is legal if the source isn't a write.
    if (!Src->mayWriteToMemory())
      return true;

    // At least one of the accesses must be strided.
    if (!isStrided(SrcDes.Stride) && !isStrided(SinkDes.Stride))
      return true;

    // If dependence information is not available from LoopAccessInfo,
    // conservatively assume the instructions can't be reordered.
    if (!areDependencesValid())
      return false;

    // If we know there is a dependence from source to sink, assume the
    // instructions can't be reordered. Otherwise, reordering is legal.
    return Dependences.find(Src) == Dependences.end() ||
           !Dependences.lookup(Src).count(Sink);
  }

  /// Collect the dependences from LoopAccessInfo.
  ///
  /// We process the dependences once during the interleaved access analysis to
  /// enable constant-time dependence queries.
  void collectDependences() {
    if (!areDependencesValid())
      return;
    auto *Deps = LAI->getDepChecker().getDependences();
    for (auto Dep : *Deps)
      Dependences[Dep.getSource(*LAI)].insert(Dep.getDestination(*LAI));
  }
};

} // llvm namespace

#endif
