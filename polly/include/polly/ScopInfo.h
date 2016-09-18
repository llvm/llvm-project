//===------ polly/ScopInfo.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Store the polyhedral model representation of a static control flow region,
// also called SCoP (Static Control Part).
//
// This representation is shared among several tools in the polyhedral
// community, which are e.g. CLooG, Pluto, Loopo, Graphite.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_INFO_H
#define POLLY_SCOP_INFO_H

#include "polly/ScopDetection.h"
#include "polly/Support/SCEVAffinator.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/RegionPass.h"
#include "isl/aff.h"
#include "isl/ctx.h"
#include "isl/set.h"

#include <deque>
#include <forward_list>

using namespace llvm;

namespace llvm {
class AssumptionCache;
class Loop;
class LoopInfo;
class PHINode;
class ScalarEvolution;
class SCEV;
class SCEVAddRecExpr;
class Type;
} // namespace llvm

struct isl_ctx;
struct isl_map;
struct isl_basic_map;
struct isl_id;
struct isl_set;
struct isl_union_set;
struct isl_union_map;
struct isl_space;
struct isl_ast_build;
struct isl_constraint;
struct isl_pw_aff;
struct isl_pw_multi_aff;
struct isl_schedule;

namespace polly {

class MemoryAccess;
class Scop;
class ScopStmt;
class ScopBuilder;

//===---------------------------------------------------------------------===//

/// Enumeration of assumptions Polly can take.
enum AssumptionKind {
  ALIASING,
  INBOUNDS,
  WRAPPING,
  UNSIGNED,
  PROFITABLE,
  ERRORBLOCK,
  COMPLEXITY,
  INFINITELOOP,
  INVARIANTLOAD,
  DELINEARIZATION,
};

/// Enum to distinguish between assumptions and restrictions.
enum AssumptionSign { AS_ASSUMPTION, AS_RESTRICTION };

/// Maps from a loop to the affine function expressing its backedge taken count.
/// The backedge taken count already enough to express iteration domain as we
/// only allow loops with canonical induction variable.
/// A canonical induction variable is:
/// an integer recurrence that starts at 0 and increments by one each time
/// through the loop.
typedef std::map<const Loop *, const SCEV *> LoopBoundMapType;

typedef std::vector<std::unique_ptr<MemoryAccess>> AccFuncVector;

/// A class to store information about arrays in the SCoP.
///
/// Objects are accessible via the ScoP, MemoryAccess or the id associated with
/// the MemoryAccess access function.
///
class ScopArrayInfo {
public:
  /// The kind of a ScopArrayInfo memory object.
  ///
  /// We distinguish between arrays and various scalar memory objects. We use
  /// the term ``array'' to describe memory objects that consist of a set of
  /// individual data elements arranged in a multi-dimensional grid. A scalar
  /// memory object describes an individual data element and is used to model
  /// the definition and uses of llvm::Values.
  ///
  /// The polyhedral model does traditionally not reason about SSA values. To
  /// reason about llvm::Values we model them "as if" they were zero-dimensional
  /// memory objects, even though they were not actually allocated in (main)
  /// memory.  Memory for such objects is only alloca[ed] at CodeGeneration
  /// time. To relate the memory slots used during code generation with the
  /// llvm::Values they belong to the new names for these corresponding stack
  /// slots are derived by appending suffixes (currently ".s2a" and ".phiops")
  /// to the name of the original llvm::Value. To describe how def/uses are
  /// modeled exactly we use these suffixes here as well.
  ///
  /// There are currently four different kinds of memory objects:
  enum MemoryKind {
    /// MK_Array: Models a one or multi-dimensional array
    ///
    /// A memory object that can be described by a multi-dimensional array.
    /// Memory objects of this type are used to model actual multi-dimensional
    /// arrays as they exist in LLVM-IR, but they are also used to describe
    /// other objects:
    ///   - A single data element allocated on the stack using 'alloca' is
    ///     modeled as a one-dimensional, single-element array.
    ///   - A single data element allocated as a global variable is modeled as
    ///     one-dimensional, single-element array.
    ///   - Certain multi-dimensional arrays with variable size, which in
    ///     LLVM-IR are commonly expressed as a single-dimensional access with a
    ///     complicated access function, are modeled as multi-dimensional
    ///     memory objects (grep for "delinearization").
    MK_Array,

    /// MK_Value: Models an llvm::Value
    ///
    /// Memory objects of type MK_Value are used to model the data flow
    /// induced by llvm::Values. For each llvm::Value that is used across
    /// BasicBocks one ScopArrayInfo object is created. A single memory WRITE
    /// stores the llvm::Value at its definition into the memory object and at
    /// each use of the llvm::Value (ignoring trivial intra-block uses) a
    /// corresponding READ is added. For instance, the use/def chain of a
    /// llvm::Value %V depicted below
    ///              ______________________
    ///              |DefBB:              |
    ///              |  %V = float op ... |
    ///              ----------------------
    ///               |                  |
    /// _________________               _________________
    /// |UseBB1:        |               |UseBB2:        |
    /// |  use float %V |               |  use float %V |
    /// -----------------               -----------------
    ///
    /// is modeled as if the following memory accesses occured:
    ///
    ///                        __________________________
    ///                        |entry:                  |
    ///                        |  %V.s2a = alloca float |
    ///                        --------------------------
    ///                                     |
    ///                    ___________________________________
    ///                    |DefBB:                           |
    ///                    |  store %float %V, float* %V.s2a |
    ///                    -----------------------------------
    ///                           |                   |
    /// ____________________________________ ___________________________________
    /// |UseBB1:                           | |UseBB2:                          |
    /// |  %V.reload1 = load float* %V.s2a | |  %V.reload2 = load float* %V.s2a|
    /// |  use float %V.reload1            | |  use float %V.reload2           |
    /// ------------------------------------ -----------------------------------
    ///
    MK_Value,

    /// MK_PHI: Models PHI nodes within the SCoP
    ///
    /// Besides the MK_Value memory object used to model the normal
    /// llvm::Value dependences described above, PHI nodes require an additional
    /// memory object of type MK_PHI to describe the forwarding of values to
    /// the PHI node.
    ///
    /// As an example, a PHIInst instructions
    ///
    /// %PHI = phi float [ %Val1, %IncomingBlock1 ], [ %Val2, %IncomingBlock2 ]
    ///
    /// is modeled as if the accesses occured this way:
    ///
    ///                    _______________________________
    ///                    |entry:                       |
    ///                    |  %PHI.phiops = alloca float |
    ///                    -------------------------------
    ///                           |              |
    /// __________________________________  __________________________________
    /// |IncomingBlock1:                 |  |IncomingBlock2:                 |
    /// |  ...                           |  |  ...                           |
    /// |  store float %Val1 %PHI.phiops |  |  store float %Val2 %PHI.phiops |
    /// |  br label % JoinBlock          |  |  br label %JoinBlock           |
    /// ----------------------------------  ----------------------------------
    ///                             \            /
    ///                              \          /
    ///               _________________________________________
    ///               |JoinBlock:                             |
    ///               |  %PHI = load float, float* PHI.phiops |
    ///               -----------------------------------------
    ///
    /// Note that there can also be a scalar write access for %PHI if used in a
    /// different BasicBlock, i.e. there can be a memory object %PHI.phiops as
    /// well as a memory object %PHI.s2a.
    MK_PHI,

    /// MK_ExitPHI: Models PHI nodes in the SCoP's exit block
    ///
    /// For PHI nodes in the Scop's exit block a special memory object kind is
    /// used. The modeling used is identical to MK_PHI, with the exception
    /// that there are no READs from these memory objects. The PHINode's
    /// llvm::Value is treated as a value escaping the SCoP. WRITE accesses
    /// write directly to the escaping value's ".s2a" alloca.
    MK_ExitPHI
  };

  /// Construct a ScopArrayInfo object.
  ///
  /// @param BasePtr        The array base pointer.
  /// @param ElementType    The type of the elements stored in the array.
  /// @param IslCtx         The isl context used to create the base pointer id.
  /// @param DimensionSizes A vector containing the size of each dimension.
  /// @param Kind           The kind of the array object.
  /// @param DL             The data layout of the module.
  /// @param S              The scop this array object belongs to.
  /// @param BaseName       The optional name of this memory reference.
  ScopArrayInfo(Value *BasePtr, Type *ElementType, isl_ctx *IslCtx,
                ArrayRef<const SCEV *> DimensionSizes, enum MemoryKind Kind,
                const DataLayout &DL, Scop *S, const char *BaseName = nullptr);

  ///  Update the element type of the ScopArrayInfo object.
  ///
  ///  Memory accesses referencing this ScopArrayInfo object may use
  ///  different element sizes. This function ensures the canonical element type
  ///  stored is small enough to model accesses to the current element type as
  ///  well as to @p NewElementType.
  ///
  ///  @param NewElementType An element type that is used to access this array.
  void updateElementType(Type *NewElementType);

  ///  Update the sizes of the ScopArrayInfo object.
  ///
  ///  A ScopArrayInfo object may be created without all outer dimensions being
  ///  available. This function is called when new memory accesses are added for
  ///  this ScopArrayInfo object. It verifies that sizes are compatible and adds
  ///  additional outer array dimensions, if needed.
  ///
  ///  @param Sizes       A vector of array sizes where the rightmost array
  ///                     sizes need to match the innermost array sizes already
  ///                     defined in SAI.
  bool updateSizes(ArrayRef<const SCEV *> Sizes);

  /// Destructor to free the isl id of the base pointer.
  ~ScopArrayInfo();

  /// Set the base pointer to @p BP.
  void setBasePtr(Value *BP) { BasePtr = BP; }

  /// Return the base pointer.
  Value *getBasePtr() const { return BasePtr; }

  /// For indirect accesses return the origin SAI of the BP, else null.
  const ScopArrayInfo *getBasePtrOriginSAI() const { return BasePtrOriginSAI; }

  /// The set of derived indirect SAIs for this origin SAI.
  const SmallPtrSetImpl<ScopArrayInfo *> &getDerivedSAIs() const {
    return DerivedSAIs;
  }

  /// Return the number of dimensions.
  unsigned getNumberOfDimensions() const {
    if (Kind == MK_PHI || Kind == MK_ExitPHI || Kind == MK_Value)
      return 0;
    return DimensionSizes.size();
  }

  /// Return the size of dimension @p dim as SCEV*.
  //
  //  Scalars do not have array dimensions and the first dimension of
  //  a (possibly multi-dimensional) array also does not carry any size
  //  information, in case the array is not newly created.
  const SCEV *getDimensionSize(unsigned Dim) const {
    assert(Dim < getNumberOfDimensions() && "Invalid dimension");
    return DimensionSizes[Dim];
  }

  /// Return the size of dimension @p dim as isl_pw_aff.
  //
  //  Scalars do not have array dimensions and the first dimension of
  //  a (possibly multi-dimensional) array also does not carry any size
  //  information, in case the array is not newly created.
  __isl_give isl_pw_aff *getDimensionSizePw(unsigned Dim) const {
    assert(Dim < getNumberOfDimensions() && "Invalid dimension");
    return isl_pw_aff_copy(DimensionSizesPw[Dim]);
  }

  /// Get the canonical element type of this array.
  ///
  /// @returns The canonical element type of this array.
  Type *getElementType() const { return ElementType; }

  /// Get element size in bytes.
  int getElemSizeInBytes() const;

  /// Get the name of this memory reference.
  std::string getName() const;

  /// Return the isl id for the base pointer.
  __isl_give isl_id *getBasePtrId() const;

  /// Return what kind of memory this represents.
  enum MemoryKind getKind() const { return Kind; }

  /// Is this array info modeling an llvm::Value?
  bool isValueKind() const { return Kind == MK_Value; }

  /// Is this array info modeling special PHI node memory?
  ///
  /// During code generation of PHI nodes, there is a need for two kinds of
  /// virtual storage. The normal one as it is used for all scalar dependences,
  /// where the result of the PHI node is stored and later loaded from as well
  /// as a second one where the incoming values of the PHI nodes are stored
  /// into and reloaded when the PHI is executed. As both memories use the
  /// original PHI node as virtual base pointer, we have this additional
  /// attribute to distinguish the PHI node specific array modeling from the
  /// normal scalar array modeling.
  bool isPHIKind() const { return Kind == MK_PHI; }

  /// Is this array info modeling an MK_ExitPHI?
  bool isExitPHIKind() const { return Kind == MK_ExitPHI; }

  /// Is this array info modeling an array?
  bool isArrayKind() const { return Kind == MK_Array; }

  /// Dump a readable representation to stderr.
  void dump() const;

  /// Print a readable representation to @p OS.
  ///
  /// @param SizeAsPwAff Print the size as isl_pw_aff
  void print(raw_ostream &OS, bool SizeAsPwAff = false) const;

  /// Access the ScopArrayInfo associated with an access function.
  static const ScopArrayInfo *
  getFromAccessFunction(__isl_keep isl_pw_multi_aff *PMA);

  /// Access the ScopArrayInfo associated with an isl Id.
  static const ScopArrayInfo *getFromId(__isl_take isl_id *Id);

  /// Get the space of this array access.
  __isl_give isl_space *getSpace() const;

  /// If the array is read only
  bool isReadOnly();

private:
  void addDerivedSAI(ScopArrayInfo *DerivedSAI) {
    DerivedSAIs.insert(DerivedSAI);
  }

  /// For indirect accesses this is the SAI of the BP origin.
  const ScopArrayInfo *BasePtrOriginSAI;

  /// For origin SAIs the set of derived indirect SAIs.
  SmallPtrSet<ScopArrayInfo *, 2> DerivedSAIs;

  /// The base pointer.
  AssertingVH<Value> BasePtr;

  /// The canonical element type of this array.
  ///
  /// The canonical element type describes the minimal accessible element in
  /// this array. Not all elements accessed, need to be of the very same type,
  /// but the allocation size of the type of the elements loaded/stored from/to
  /// this array needs to be a multiple of the allocation size of the canonical
  /// type.
  Type *ElementType;

  /// The isl id for the base pointer.
  isl_id *Id;

  /// The sizes of each dimension as SCEV*.
  SmallVector<const SCEV *, 4> DimensionSizes;

  /// The sizes of each dimension as isl_pw_aff.
  SmallVector<isl_pw_aff *, 4> DimensionSizesPw;

  /// The type of this scop array info object.
  ///
  /// We distinguish between SCALAR, PHI and ARRAY objects.
  enum MemoryKind Kind;

  /// The data layout of the module.
  const DataLayout &DL;

  /// The scop this SAI object belongs to.
  Scop &S;
};

/// Represent memory accesses in statements.
class MemoryAccess {
  friend class Scop;
  friend class ScopStmt;

public:
  /// The access type of a memory access
  ///
  /// There are three kind of access types:
  ///
  /// * A read access
  ///
  /// A certain set of memory locations are read and may be used for internal
  /// calculations.
  ///
  /// * A must-write access
  ///
  /// A certain set of memory locations is definitely written. The old value is
  /// replaced by a newly calculated value. The old value is not read or used at
  /// all.
  ///
  /// * A may-write access
  ///
  /// A certain set of memory locations may be written. The memory location may
  /// contain a new value if there is actually a write or the old value may
  /// remain, if no write happens.
  enum AccessType {
    READ = 0x1,
    MUST_WRITE = 0x2,
    MAY_WRITE = 0x3,
  };

  /// Reduction access type
  ///
  /// Commutative and associative binary operations suitable for reductions
  enum ReductionType {
    RT_NONE, ///< Indicate no reduction at all
    RT_ADD,  ///< Addition
    RT_MUL,  ///< Multiplication
    RT_BOR,  ///< Bitwise Or
    RT_BXOR, ///< Bitwise XOr
    RT_BAND, ///< Bitwise And
  };

private:
  MemoryAccess(const MemoryAccess &) = delete;
  const MemoryAccess &operator=(const MemoryAccess &) = delete;

  /// A unique identifier for this memory access.
  ///
  /// The identifier is unique between all memory accesses belonging to the same
  /// scop statement.
  isl_id *Id;

  /// What is modeled by this MemoryAccess.
  /// @see ScopArrayInfo::MemoryKind
  ScopArrayInfo::MemoryKind Kind;

  /// Whether it a reading or writing access, and if writing, whether it
  /// is conditional (MAY_WRITE).
  enum AccessType AccType;

  /// Reduction type for reduction like accesses, RT_NONE otherwise
  ///
  /// An access is reduction like if it is part of a load-store chain in which
  /// both access the same memory location (use the same LLVM-IR value
  /// as pointer reference). Furthermore, between the load and the store there
  /// is exactly one binary operator which is known to be associative and
  /// commutative.
  ///
  /// TODO:
  ///
  /// We can later lift the constraint that the same LLVM-IR value defines the
  /// memory location to handle scops such as the following:
  ///
  ///    for i
  ///      for j
  ///        sum[i+j] = sum[i] + 3;
  ///
  /// Here not all iterations access the same memory location, but iterations
  /// for which j = 0 holds do. After lifting the equality check in ScopBuilder,
  /// subsequent transformations do not only need check if a statement is
  /// reduction like, but they also need to verify that that the reduction
  /// property is only exploited for statement instances that load from and
  /// store to the same data location. Doing so at dependence analysis time
  /// could allow us to handle the above example.
  ReductionType RedType = RT_NONE;

  /// Parent ScopStmt of this access.
  ScopStmt *Statement;

  /// The domain under which this access is not modeled precisely.
  ///
  /// The invalid domain for an access describes all parameter combinations
  /// under which the statement looks to be executed but is in fact not because
  /// some assumption/restriction makes the access invalid.
  isl_set *InvalidDomain;

  // Properties describing the accessed array.
  // TODO: It might be possible to move them to ScopArrayInfo.
  // @{

  /// The base address (e.g., A for A[i+j]).
  ///
  /// The #BaseAddr of a memory access of kind MK_Array is the base pointer
  /// of the memory access.
  /// The #BaseAddr of a memory access of kind MK_PHI or MK_ExitPHI is the
  /// PHI node itself.
  /// The #BaseAddr of a memory access of kind MK_Value is the instruction
  /// defining the value.
  AssertingVH<Value> BaseAddr;

  /// An unique name of the accessed array.
  std::string BaseName;

  /// Type a single array element wrt. this access.
  Type *ElementType;

  /// Size of each dimension of the accessed array.
  SmallVector<const SCEV *, 4> Sizes;
  // @}

  // Properties describing the accessed element.
  // @{

  /// The access instruction of this memory access.
  ///
  /// For memory accesses of kind MK_Array the access instruction is the
  /// Load or Store instruction performing the access.
  ///
  /// For memory accesses of kind MK_PHI or MK_ExitPHI the access
  /// instruction of a load access is the PHI instruction. The access
  /// instruction of a PHI-store is the incoming's block's terminator
  /// instruction.
  ///
  /// For memory accesses of kind MK_Value the access instruction of a load
  /// access is nullptr because generally there can be multiple instructions in
  /// the statement using the same llvm::Value. The access instruction of a
  /// write access is the instruction that defines the llvm::Value.
  Instruction *AccessInstruction;

  /// Incoming block and value of a PHINode.
  SmallVector<std::pair<BasicBlock *, Value *>, 4> Incoming;

  /// The value associated with this memory access.
  ///
  ///  - For array memory accesses (MK_Array) it is the loaded result or the
  ///    stored value. If the access instruction is a memory intrinsic it
  ///    the access value is also the memory intrinsic.
  ///  - For accesses of kind MK_Value it is the access instruction itself.
  ///  - For accesses of kind MK_PHI or MK_ExitPHI it is the PHI node itself
  ///    (for both, READ and WRITE accesses).
  ///
  AssertingVH<Value> AccessValue;

  /// Are all the subscripts affine expression?
  bool IsAffine;

  /// Subscript expression for each dimension.
  SmallVector<const SCEV *, 4> Subscripts;

  /// Relation from statement instances to the accessed array elements.
  ///
  /// In the common case this relation is a function that maps a set of loop
  /// indices to the memory address from which a value is loaded/stored:
  ///
  ///      for i
  ///        for j
  ///    S:     A[i + 3 j] = ...
  ///
  ///    => { S[i,j] -> A[i + 3j] }
  ///
  /// In case the exact access function is not known, the access relation may
  /// also be a one to all mapping { S[i,j] -> A[o] } describing that any
  /// element accessible through A might be accessed.
  ///
  /// In case of an access to a larger element belonging to an array that also
  /// contains smaller elements, the access relation models the larger access
  /// with multiple smaller accesses of the size of the minimal array element
  /// type:
  ///
  ///      short *A;
  ///
  ///      for i
  ///    S:     A[i] = *((double*)&A[4 * i]);
  ///
  ///    => { S[i] -> A[i]; S[i] -> A[o] : 4i <= o <= 4i + 3 }
  isl_map *AccessRelation;

  /// Updated access relation read from JSCOP file.
  isl_map *NewAccessRelation;
  // @}

  bool isAffine() const { return IsAffine; }

  __isl_give isl_basic_map *createBasicAccessMap(ScopStmt *Statement);

  void assumeNoOutOfBound();

  /// Compute bounds on an over approximated  access relation.
  ///
  /// @param ElementSize The size of one element accessed.
  void computeBoundsOnAccessRelation(unsigned ElementSize);

  /// Get the original access function as read from IR.
  __isl_give isl_map *getOriginalAccessRelation() const;

  /// Return the space in which the access relation lives in.
  __isl_give isl_space *getOriginalAccessRelationSpace() const;

  /// Get the new access function imported or set by a pass
  __isl_give isl_map *getNewAccessRelation() const;

  /// Fold the memory access to consider parameteric offsets
  ///
  /// To recover memory accesses with array size parameters in the subscript
  /// expression we post-process the delinearization results.
  ///
  /// We would normally recover from an access A[exp0(i) * N + exp1(i)] into an
  /// array A[][N] the 2D access A[exp0(i)][exp1(i)]. However, another valid
  /// delinearization is A[exp0(i) - 1][exp1(i) + N] which - depending on the
  /// range of exp1(i) - may be preferrable. Specifically, for cases where we
  /// know exp1(i) is negative, we want to choose the latter expression.
  ///
  /// As we commonly do not have any information about the range of exp1(i),
  /// we do not choose one of the two options, but instead create a piecewise
  /// access function that adds the (-1, N) offsets as soon as exp1(i) becomes
  /// negative. For a 2D array such an access function is created by applying
  /// the piecewise map:
  ///
  /// [i,j] -> [i, j] :      j >= 0
  /// [i,j] -> [i-1, j+N] :  j <  0
  ///
  /// We can generalize this mapping to arbitrary dimensions by applying this
  /// piecewise mapping pairwise from the rightmost to the leftmost access
  /// dimension. It would also be possible to cover a wider range by introducing
  /// more cases and adding multiple of Ns to these cases. However, this has
  /// not yet been necessary.
  /// The introduction of different cases necessarily complicates the memory
  /// access function, but cases that can be statically proven to not happen
  /// will be eliminated later on.
  __isl_give isl_map *foldAccess(__isl_take isl_map *AccessRelation,
                                 ScopStmt *Statement);

  /// Create the access relation for the underlying memory intrinsic.
  void buildMemIntrinsicAccessRelation();

  /// Assemble the access relation from all available information.
  ///
  /// In particular, used the information passes in the constructor and the
  /// parent ScopStmt set by setStatment().
  ///
  /// @param SAI Info object for the accessed array.
  void buildAccessRelation(const ScopArrayInfo *SAI);

  /// Carry index overflows of dimensions with constant size to the next higher
  /// dimension.
  ///
  /// For dimensions that have constant size, modulo the index by the size and
  /// add up the carry (floored division) to the next higher dimension. This is
  /// how overflow is defined in row-major order.
  /// It happens e.g. when ScalarEvolution computes the offset to the base
  /// pointer and would algebraically sum up all lower dimensions' indices of
  /// constant size.
  ///
  /// Example:
  ///   float (*A)[4];
  ///   A[1][6] -> A[2][2]
  void wrapConstantDimensions();

public:
  /// Create a new MemoryAccess.
  ///
  /// @param Stmt       The parent statement.
  /// @param AccessInst The instruction doing the access.
  /// @param BaseAddr   The accessed array's address.
  /// @param ElemType   The type of the accessed array elements.
  /// @param AccType    Whether read or write access.
  /// @param IsAffine   Whether the subscripts are affine expressions.
  /// @param Kind       The kind of memory accessed.
  /// @param Subscripts Subscipt expressions
  /// @param Sizes      Dimension lengths of the accessed array.
  /// @param BaseName   Name of the acessed array.
  MemoryAccess(ScopStmt *Stmt, Instruction *AccessInst, AccessType AccType,
               Value *BaseAddress, Type *ElemType, bool Affine,
               ArrayRef<const SCEV *> Subscripts, ArrayRef<const SCEV *> Sizes,
               Value *AccessValue, ScopArrayInfo::MemoryKind Kind,
               StringRef BaseName);

  /// Create a new MemoryAccess that corresponds to @p AccRel.
  ///
  /// Along with @p Stmt and @p AccType it uses information about dimension
  /// lengths of the accessed array, the type of the accessed array elements,
  /// the name of the accessed array that is derived from the object accessible
  /// via @p AccRel.
  ///
  /// @param Stmt       The parent statement.
  /// @param AccType    Whether read or write access.
  /// @param AccRel     The access relation that describes the memory access.
  MemoryAccess(ScopStmt *Stmt, AccessType AccType, __isl_take isl_map *AccRel);

  ~MemoryAccess();

  /// Add a new incoming block/value pairs for this PHI/ExitPHI access.
  ///
  /// @param IncomingBlock The PHI's incoming block.
  /// @param IncomingValue The value when reacing the PHI from the @p
  ///                      IncomingBlock.
  void addIncoming(BasicBlock *IncomingBlock, Value *IncomingValue) {
    assert(!isRead());
    assert(isAnyPHIKind());
    Incoming.emplace_back(std::make_pair(IncomingBlock, IncomingValue));
  }

  /// Return the list of possible PHI/ExitPHI values.
  ///
  /// After code generation moves some PHIs around during region simplification,
  /// we cannot reliably locate the original PHI node and its incoming values
  /// anymore. For this reason we remember these explicitly for all PHI-kind
  /// accesses.
  ArrayRef<std::pair<BasicBlock *, Value *>> getIncoming() const {
    assert(isAnyPHIKind());
    return Incoming;
  }

  /// Get the type of a memory access.
  enum AccessType getType() { return AccType; }

  /// Is this a reduction like access?
  bool isReductionLike() const { return RedType != RT_NONE; }

  /// Is this a read memory access?
  bool isRead() const { return AccType == MemoryAccess::READ; }

  /// Is this a must-write memory access?
  bool isMustWrite() const { return AccType == MemoryAccess::MUST_WRITE; }

  /// Is this a may-write memory access?
  bool isMayWrite() const { return AccType == MemoryAccess::MAY_WRITE; }

  /// Is this a write memory access?
  bool isWrite() const { return isMustWrite() || isMayWrite(); }

  /// Check if a new access relation was imported or set by a pass.
  bool hasNewAccessRelation() const { return NewAccessRelation; }

  /// Return the newest access relation of this access.
  ///
  /// There are two possibilities:
  ///   1) The original access relation read from the LLVM-IR.
  ///   2) A new access relation imported from a json file or set by another
  ///      pass (e.g., for privatization).
  ///
  /// As 2) is by construction "newer" than 1) we return the new access
  /// relation if present.
  ///
  __isl_give isl_map *getLatestAccessRelation() const {
    return hasNewAccessRelation() ? getNewAccessRelation()
                                  : getOriginalAccessRelation();
  }

  /// Old name of getLatestAccessRelation().
  __isl_give isl_map *getAccessRelation() const {
    return getLatestAccessRelation();
  }

  /// Get an isl map describing the memory address accessed.
  ///
  /// In most cases the memory address accessed is well described by the access
  /// relation obtained with getAccessRelation. However, in case of arrays
  /// accessed with types of different size the access relation maps one access
  /// to multiple smaller address locations. This method returns an isl map that
  /// relates each dynamic statement instance to the unique memory location
  /// that is loaded from / stored to.
  ///
  /// For an access relation { S[i] -> A[o] : 4i <= o <= 4i + 3 } this method
  /// will return the address function { S[i] -> A[4i] }.
  ///
  /// @returns The address function for this memory access.
  __isl_give isl_map *getAddressFunction() const;

  /// Return the access relation after the schedule was applied.
  __isl_give isl_pw_multi_aff *
  applyScheduleToAccessRelation(__isl_take isl_union_map *Schedule) const;

  /// Get an isl string representing the access function read from IR.
  std::string getOriginalAccessRelationStr() const;

  /// Get an isl string representing a new access function, if available.
  std::string getNewAccessRelationStr() const;

  /// Get the base address of this access (e.g. A for A[i+j]) when
  /// detected.
  Value *getOriginalBaseAddr() const {
    assert(!getOriginalScopArrayInfo() /* may noy yet be initialized */ ||
           getOriginalScopArrayInfo()->getBasePtr() == BaseAddr);
    return BaseAddr;
  }

  /// Get the base address of this access (e.g. A for A[i+j]) after a
  /// potential change by setNewAccessRelation().
  Value *getLatestBaseAddr() const {
    return getLatestScopArrayInfo()->getBasePtr();
  }

  /// Old name for getOriginalBaseAddr().
  Value *getBaseAddr() const { return getOriginalBaseAddr(); }

  /// Get the detection-time base array isl_id for this access.
  __isl_give isl_id *getOriginalArrayId() const;

  /// Get the base array isl_id for this access, modifiable through
  /// setNewAccessRelation().
  __isl_give isl_id *getLatestArrayId() const;

  /// Old name of getOriginalArrayId().
  __isl_give isl_id *getArrayId() const { return getOriginalArrayId(); }

  /// Get the detection-time ScopArrayInfo object for the base address.
  const ScopArrayInfo *getOriginalScopArrayInfo() const;

  /// Get the ScopArrayInfo object for the base address, or the one set
  /// by setNewAccessRelation().
  const ScopArrayInfo *getLatestScopArrayInfo() const;

  /// Legacy name of getOriginalScopArrayInfo().
  const ScopArrayInfo *getScopArrayInfo() const {
    return getOriginalScopArrayInfo();
  }

  /// Return a string representation of the access's reduction type.
  const std::string getReductionOperatorStr() const;

  /// Return a string representation of the reduction type @p RT.
  static const std::string getReductionOperatorStr(ReductionType RT);

  const std::string &getBaseName() const { return BaseName; }

  /// Return the element type of the accessed array wrt. this access.
  Type *getElementType() const { return ElementType; }

  /// Return the access value of this memory access.
  Value *getAccessValue() const { return AccessValue; }

  /// Return the access instruction of this memory access.
  Instruction *getAccessInstruction() const { return AccessInstruction; }

  /// Return the number of access function subscript.
  unsigned getNumSubscripts() const { return Subscripts.size(); }

  /// Return the access function subscript in the dimension @p Dim.
  const SCEV *getSubscript(unsigned Dim) const { return Subscripts[Dim]; }

  /// Compute the isl representation for the SCEV @p E wrt. this access.
  ///
  /// Note that this function will also adjust the invalid context accordingly.
  __isl_give isl_pw_aff *getPwAff(const SCEV *E);

  /// Get the invalid domain for this access.
  __isl_give isl_set *getInvalidDomain() const {
    return isl_set_copy(InvalidDomain);
  }

  /// Get the invalid context for this access.
  __isl_give isl_set *getInvalidContext() const {
    return isl_set_params(getInvalidDomain());
  }

  /// Get the stride of this memory access in the specified Schedule. Schedule
  /// is a map from the statement to a schedule where the innermost dimension is
  /// the dimension of the innermost loop containing the statement.
  __isl_give isl_set *getStride(__isl_take const isl_map *Schedule) const;

  /// Is the stride of the access equal to a certain width? Schedule is a map
  /// from the statement to a schedule where the innermost dimension is the
  /// dimension of the innermost loop containing the statement.
  bool isStrideX(__isl_take const isl_map *Schedule, int StrideWidth) const;

  /// Is consecutive memory accessed for a given statement instance set?
  /// Schedule is a map from the statement to a schedule where the innermost
  /// dimension is the dimension of the innermost loop containing the
  /// statement.
  bool isStrideOne(__isl_take const isl_map *Schedule) const;

  /// Is always the same memory accessed for a given statement instance set?
  /// Schedule is a map from the statement to a schedule where the innermost
  /// dimension is the dimension of the innermost loop containing the
  /// statement.
  bool isStrideZero(__isl_take const isl_map *Schedule) const;

  /// Return the kind when this access was first detected.
  ScopArrayInfo::MemoryKind getOriginalKind() const {
    assert(!getOriginalScopArrayInfo() /* not yet initialized */ ||
           getOriginalScopArrayInfo()->getKind() == Kind);
    return Kind;
  }

  /// Return the kind considering a potential setNewAccessRelation.
  ScopArrayInfo::MemoryKind getLatestKind() const {
    return getLatestScopArrayInfo()->getKind();
  }

  /// Whether this is an access of an explicit load or store in the IR.
  bool isOriginalArrayKind() const {
    return getOriginalKind() == ScopArrayInfo::MK_Array;
  }

  /// Whether storage memory is either an custom .s2a/.phiops alloca
  /// (false) or an existing pointer into an array (true).
  bool isLatestArrayKind() const {
    return getLatestKind() == ScopArrayInfo::MK_Array;
  }

  /// Old name of isOriginalArrayKind.
  bool isArrayKind() const { return isOriginalArrayKind(); }

  /// Whether this access is an array to a scalar memory object, without
  /// considering changes by setNewAccessRelation.
  ///
  /// Scalar accesses are accesses to MK_Value, MK_PHI or MK_ExitPHI.
  bool isOriginalScalarKind() const {
    return getOriginalKind() != ScopArrayInfo::MK_Array;
  }

  /// Whether this access is an array to a scalar memory object, also
  /// considering changes by setNewAccessRelation.
  bool isLatestScalarKind() const {
    return getLatestKind() != ScopArrayInfo::MK_Array;
  }

  /// Old name of isOriginalScalarKind.
  bool isScalarKind() const { return isOriginalScalarKind(); }

  /// Was this MemoryAccess detected as a scalar dependences?
  bool isOriginalValueKind() const {
    return getOriginalKind() == ScopArrayInfo::MK_Value;
  }

  /// Is this MemoryAccess currently modeling scalar dependences?
  bool isLatestValueKind() const {
    return getLatestKind() == ScopArrayInfo::MK_Value;
  }

  /// Old name of isOriginalValueKind().
  bool isValueKind() const { return isOriginalValueKind(); }

  /// Was this MemoryAccess detected as a special PHI node access?
  bool isOriginalPHIKind() const {
    return getOriginalKind() == ScopArrayInfo::MK_PHI;
  }

  /// Is this MemoryAccess modeling special PHI node accesses, also
  /// considering a potential change by setNewAccessRelation?
  bool isLatestPHIKind() const {
    return getLatestKind() == ScopArrayInfo::MK_PHI;
  }

  /// Old name of isOriginalPHIKind.
  bool isPHIKind() const { return isOriginalPHIKind(); }

  /// Was this MemoryAccess detected as the accesses of a PHI node in the
  /// SCoP's exit block?
  bool isOriginalExitPHIKind() const {
    return getOriginalKind() == ScopArrayInfo::MK_ExitPHI;
  }

  /// Is this MemoryAccess modeling the accesses of a PHI node in the
  /// SCoP's exit block? Can be changed to an array access using
  /// setNewAccessRelation().
  bool isLatestExitPHIKind() const {
    return getLatestKind() == ScopArrayInfo::MK_ExitPHI;
  }

  /// Old name of isOriginalExitPHIKind().
  bool isExitPHIKind() const { return isOriginalExitPHIKind(); }

  /// Was this access detected as one of the two PHI types?
  bool isOriginalAnyPHIKind() const {
    return isOriginalPHIKind() || isOriginalExitPHIKind();
  }

  /// Does this access orginate from one of the two PHI types? Can be
  /// changed to an array access using setNewAccessRelation().
  bool isLatestAnyPHIKind() const {
    return isLatestPHIKind() || isLatestExitPHIKind();
  }

  /// Old name of isOriginalAnyPHIKind().
  bool isAnyPHIKind() const { return isOriginalAnyPHIKind(); }

  /// Get the statement that contains this memory access.
  ScopStmt *getStatement() const { return Statement; }

  /// Get the reduction type of this access
  ReductionType getReductionType() const { return RedType; }

  /// Set the updated access relation read from JSCOP file.
  void setNewAccessRelation(__isl_take isl_map *NewAccessRelation);

  /// Mark this a reduction like access
  void markAsReductionLike(ReductionType RT) { RedType = RT; }

  /// Align the parameters in the access relation to the scop context
  void realignParams();

  /// Update the dimensionality of the memory access.
  ///
  /// During scop construction some memory accesses may not be constructed with
  /// their full dimensionality, but outer dimensions may have been omitted if
  /// they took the value 'zero'. By updating the dimensionality of the
  /// statement we add additional zero-valued dimensions to match the
  /// dimensionality of the ScopArrayInfo object that belongs to this memory
  /// access.
  void updateDimensionality();

  /// Get identifier for the memory access.
  ///
  /// This identifier is unique for all accesses that belong to the same scop
  /// statement.
  __isl_give isl_id *getId() const;

  /// Print the MemoryAccess.
  ///
  /// @param OS The output stream the MemoryAccess is printed to.
  void print(raw_ostream &OS) const;

  /// Print the MemoryAccess to stderr.
  void dump() const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              MemoryAccess::ReductionType RT);

/// Ordered list type to hold accesses.
using MemoryAccessList = std::forward_list<MemoryAccess *>;

/// Helper structure for invariant memory accesses.
struct InvariantAccess {
  /// The memory access that is (partially) invariant.
  MemoryAccess *MA;

  /// The context under which the access is not invariant.
  isl_set *NonHoistableCtx;
};

/// Ordered container type to hold invariant accesses.
using InvariantAccessesTy = SmallVector<InvariantAccess, 8>;

/// Type for equivalent invariant accesses and their domain context.
struct InvariantEquivClassTy {

  /// The pointer that identifies this equivalence class
  const SCEV *IdentifyingPointer;

  /// Memory accesses now treated invariant
  ///
  /// These memory accesses access the pointer location that identifies
  /// this equivalence class. They are treated as invariant and hoisted during
  /// code generation.
  MemoryAccessList InvariantAccesses;

  /// The execution context under which the memory location is accessed
  ///
  /// It is the union of the execution domains of the memory accesses in the
  /// InvariantAccesses list.
  isl_set *ExecutionContext;

  /// The type of the invariant access
  ///
  /// It is used to differentiate between differently typed invariant loads from
  /// the same location.
  Type *AccessType;
};

/// Type for invariant accesses equivalence classes.
using InvariantEquivClassesTy = SmallVector<InvariantEquivClassTy, 8>;

/// Statement of the Scop
///
/// A Scop statement represents an instruction in the Scop.
///
/// It is further described by its iteration domain, its schedule and its data
/// accesses.
/// At the moment every statement represents a single basic block of LLVM-IR.
class ScopStmt {
public:
  ScopStmt(const ScopStmt &) = delete;
  const ScopStmt &operator=(const ScopStmt &) = delete;

  /// Create the ScopStmt from a BasicBlock.
  ScopStmt(Scop &parent, BasicBlock &bb);

  /// Create an overapproximating ScopStmt for the region @p R.
  ScopStmt(Scop &parent, Region &R);

  /// Create a copy statement.
  ///
  /// @param Stmt       The parent statement.
  /// @param SourceRel  The source location.
  /// @param TargetRel  The target location.
  /// @param Domain     The original domain under which copy statement whould
  ///                   be executed.
  ScopStmt(Scop &parent, __isl_take isl_map *SourceRel,
           __isl_take isl_map *TargetRel, __isl_take isl_set *Domain);

  /// Initialize members after all MemoryAccesses have been added.
  void init(LoopInfo &LI);

private:
  /// Polyhedral description
  //@{

  /// The Scop containing this ScopStmt
  Scop &Parent;

  /// The domain under which this statement is not modeled precisely.
  ///
  /// The invalid domain for a statement describes all parameter combinations
  /// under which the statement looks to be executed but is in fact not because
  /// some assumption/restriction makes the statement/scop invalid.
  isl_set *InvalidDomain;

  /// The iteration domain describes the set of iterations for which this
  /// statement is executed.
  ///
  /// Example:
  ///     for (i = 0; i < 100 + b; ++i)
  ///       for (j = 0; j < i; ++j)
  ///         S(i,j);
  ///
  /// 'S' is executed for different values of i and j. A vector of all
  /// induction variables around S (i, j) is called iteration vector.
  /// The domain describes the set of possible iteration vectors.
  ///
  /// In this case it is:
  ///
  ///     Domain: 0 <= i <= 100 + b
  ///             0 <= j <= i
  ///
  /// A pair of statement and iteration vector (S, (5,3)) is called statement
  /// instance.
  isl_set *Domain;

  /// The memory accesses of this statement.
  ///
  /// The only side effects of a statement are its memory accesses.
  typedef SmallVector<MemoryAccess *, 8> MemoryAccessVec;
  MemoryAccessVec MemAccs;

  /// Mapping from instructions to (scalar) memory accesses.
  DenseMap<const Instruction *, MemoryAccessList> InstructionToAccess;

  /// The set of values defined elsewhere required in this ScopStmt and
  ///        their MK_Value READ MemoryAccesses.
  DenseMap<Value *, MemoryAccess *> ValueReads;

  /// The set of values defined in this ScopStmt that are required
  ///        elsewhere, mapped to their MK_Value WRITE MemoryAccesses.
  DenseMap<Instruction *, MemoryAccess *> ValueWrites;

  /// Map from PHI nodes to its incoming value when coming from this
  ///        statement.
  ///
  /// Non-affine subregions can have multiple exiting blocks that are incoming
  /// blocks of the PHI nodes. This map ensures that there is only one write
  /// operation for the complete subregion. A PHI selecting the relevant value
  /// will be inserted.
  DenseMap<PHINode *, MemoryAccess *> PHIWrites;

  //@}

  /// A SCoP statement represents either a basic block (affine/precise case) or
  /// a whole region (non-affine case).
  ///
  /// Only one of the following two members will therefore be set and indicate
  /// which kind of statement this is.
  ///
  ///{

  /// The BasicBlock represented by this statement (in the affine case).
  BasicBlock *BB;

  /// The region represented by this statement (in the non-affine case).
  Region *R;

  ///}

  /// The isl AST build for the new generated AST.
  isl_ast_build *Build;

  SmallVector<Loop *, 4> NestLoops;

  std::string BaseName;

  /// Build the statement.
  //@{
  void buildDomain();

  /// Fill NestLoops with loops surrounding this statement.
  void collectSurroundingLoops();

  /// Build the access relation of all memory accesses.
  void buildAccessRelations();

  /// Detect and mark reductions in the ScopStmt
  void checkForReductions();

  /// Collect loads which might form a reduction chain with @p StoreMA
  void
  collectCandiateReductionLoads(MemoryAccess *StoreMA,
                                llvm::SmallVectorImpl<MemoryAccess *> &Loads);
  //@}

public:
  ~ScopStmt();

  /// Get an isl_ctx pointer.
  isl_ctx *getIslCtx() const;

  /// Get the iteration domain of this ScopStmt.
  ///
  /// @return The iteration domain of this ScopStmt.
  __isl_give isl_set *getDomain() const;

  /// Get the space of the iteration domain
  ///
  /// @return The space of the iteration domain
  __isl_give isl_space *getDomainSpace() const;

  /// Get the id of the iteration domain space
  ///
  /// @return The id of the iteration domain space
  __isl_give isl_id *getDomainId() const;

  /// Get an isl string representing this domain.
  std::string getDomainStr() const;

  /// Get the schedule function of this ScopStmt.
  ///
  /// @return The schedule function of this ScopStmt, if it does not contain
  /// extension nodes, and nullptr, otherwise.
  __isl_give isl_map *getSchedule() const;

  /// Get an isl string representing this schedule.
  ///
  /// @return An isl string representing this schedule, if it does not contain
  /// extension nodes, and an empty string, otherwise.
  std::string getScheduleStr() const;

  /// Get the invalid domain for this statement.
  __isl_give isl_set *getInvalidDomain() const {
    return isl_set_copy(InvalidDomain);
  }

  /// Get the invalid context for this statement.
  __isl_give isl_set *getInvalidContext() const {
    return isl_set_params(getInvalidDomain());
  }

  /// Set the invalid context for this statement to @p ID.
  void setInvalidDomain(__isl_take isl_set *ID);

  /// Get the BasicBlock represented by this ScopStmt (if any).
  ///
  /// @return The BasicBlock represented by this ScopStmt, or null if the
  ///         statement represents a region.
  BasicBlock *getBasicBlock() const { return BB; }

  /// Return true if this statement represents a single basic block.
  bool isBlockStmt() const { return BB != nullptr; }

  /// Return true if this is a copy statement.
  bool isCopyStmt() const { return BB == nullptr && R == nullptr; }

  /// Get the region represented by this ScopStmt (if any).
  ///
  /// @return The region represented by this ScopStmt, or null if the statement
  ///         represents a basic block.
  Region *getRegion() const { return R; }

  /// Return true if this statement represents a whole region.
  bool isRegionStmt() const { return R != nullptr; }

  /// Return a BasicBlock from this statement.
  ///
  /// For block statements, it returns the BasicBlock itself. For subregion
  /// statements, return its entry block.
  BasicBlock *getEntryBlock() const;

  /// Return true if this statement does not contain any accesses.
  bool isEmpty() const { return MemAccs.empty(); }

  /// Return the only array access for @p Inst, if existing.
  ///
  /// @param Inst The instruction for which to look up the access.
  /// @returns The unique array memory access related to Inst or nullptr if
  ///          no array access exists
  MemoryAccess *getArrayAccessOrNULLFor(const Instruction *Inst) const {
    auto It = InstructionToAccess.find(Inst);
    if (It == InstructionToAccess.end())
      return nullptr;

    MemoryAccess *ArrayAccess = nullptr;

    for (auto Access : It->getSecond()) {
      if (!Access->isArrayKind())
        continue;

      assert(!ArrayAccess && "More then one array access for instruction");

      ArrayAccess = Access;
    }

    return ArrayAccess;
  }

  /// Return the only array access for @p Inst.
  ///
  /// @param Inst The instruction for which to look up the access.
  /// @returns The unique array memory access related to Inst.
  MemoryAccess &getArrayAccessFor(const Instruction *Inst) const {
    MemoryAccess *ArrayAccess = getArrayAccessOrNULLFor(Inst);

    assert(ArrayAccess && "No array access found for instruction!");
    return *ArrayAccess;
  }

  /// Return the MemoryAccess that writes the value of an instruction
  ///        defined in this statement, or nullptr if not existing, respectively
  ///        not yet added.
  MemoryAccess *lookupValueWriteOf(Instruction *Inst) const {
    assert((isRegionStmt() && R->contains(Inst)) ||
           (!isRegionStmt() && Inst->getParent() == BB));
    return ValueWrites.lookup(Inst);
  }

  /// Return the MemoryAccess that reloads a value, or nullptr if not
  ///        existing, respectively not yet added.
  MemoryAccess *lookupValueReadOf(Value *Inst) const {
    return ValueReads.lookup(Inst);
  }

  /// Return the PHI write MemoryAccess for the incoming values from any
  ///        basic block in this ScopStmt, or nullptr if not existing,
  ///        respectively not yet added.
  MemoryAccess *lookupPHIWriteOf(PHINode *PHI) const {
    assert(isBlockStmt() || R->getExit() == PHI->getParent());
    return PHIWrites.lookup(PHI);
  }

  /// Add @p Access to this statement's list of accesses.
  void addAccess(MemoryAccess *Access);

  /// Remove a MemoryAccess from this statement.
  ///
  /// Note that scalar accesses that are caused by MA will
  /// be eliminated too.
  void removeMemoryAccess(MemoryAccess *MA);

  typedef MemoryAccessVec::iterator iterator;
  typedef MemoryAccessVec::const_iterator const_iterator;

  iterator begin() { return MemAccs.begin(); }
  iterator end() { return MemAccs.end(); }
  const_iterator begin() const { return MemAccs.begin(); }
  const_iterator end() const { return MemAccs.end(); }
  size_t size() const { return MemAccs.size(); }

  unsigned getNumIterators() const;

  Scop *getParent() { return &Parent; }
  const Scop *getParent() const { return &Parent; }

  const char *getBaseName() const;

  /// Set the isl AST build.
  void setAstBuild(__isl_keep isl_ast_build *B) { Build = B; }

  /// Get the isl AST build.
  __isl_keep isl_ast_build *getAstBuild() const { return Build; }

  /// Restrict the domain of the statement.
  ///
  /// @param NewDomain The new statement domain.
  void restrictDomain(__isl_take isl_set *NewDomain);

  /// Compute the isl representation for the SCEV @p E in this stmt.
  ///
  /// @param E           The SCEV that should be translated.
  /// @param NonNegative Flag to indicate the @p E has to be non-negative.
  ///
  /// Note that this function will also adjust the invalid context accordingly.
  __isl_give isl_pw_aff *getPwAff(const SCEV *E, bool NonNegative = false);

  /// Get the loop for a dimension.
  ///
  /// @param Dimension The dimension of the induction variable
  /// @return The loop at a certain dimension.
  Loop *getLoopForDimension(unsigned Dimension) const;

  /// Align the parameters in the statement to the scop context
  void realignParams();

  /// Print the ScopStmt.
  ///
  /// @param OS The output stream the ScopStmt is printed to.
  void print(raw_ostream &OS) const;

  /// Print the ScopStmt to stderr.
  void dump() const;
};

/// Print ScopStmt S to raw_ostream O.
static inline raw_ostream &operator<<(raw_ostream &O, const ScopStmt &S) {
  S.print(O);
  return O;
}

/// Static Control Part
///
/// A Scop is the polyhedral representation of a control flow region detected
/// by the Scop detection. It is generated by translating the LLVM-IR and
/// abstracting its effects.
///
/// A Scop consists of a set of:
///
///   * A set of statements executed in the Scop.
///
///   * A set of global parameters
///   Those parameters are scalar integer values, which are constant during
///   execution.
///
///   * A context
///   This context contains information about the values the parameters
///   can take and relations between different parameters.
class Scop {
public:
  /// Type to represent a pair of minimal/maximal access to an array.
  using MinMaxAccessTy = std::pair<isl_pw_multi_aff *, isl_pw_multi_aff *>;

  /// Vector of minimal/maximal accesses to different arrays.
  using MinMaxVectorTy = SmallVector<MinMaxAccessTy, 4>;

  /// Pair of minimal/maximal access vectors representing
  /// read write and read only accesses
  using MinMaxVectorPairTy = std::pair<MinMaxVectorTy, MinMaxVectorTy>;

  /// Vector of pair of minimal/maximal access vectors representing
  /// non read only and read only accesses for each alias group.
  using MinMaxVectorPairVectorTy = SmallVector<MinMaxVectorPairTy, 4>;

private:
  Scop(const Scop &) = delete;
  const Scop &operator=(const Scop &) = delete;

  ScalarEvolution *SE;

  /// The underlying Region.
  Region &R;

  // Access functions of the SCoP.
  //
  // This owns all the MemoryAccess objects of the Scop created in this pass.
  AccFuncVector AccessFunctions;

  /// Flag to indicate that the scheduler actually optimized the SCoP.
  bool IsOptimized;

  /// True if the underlying region has a single exiting block.
  bool HasSingleExitEdge;

  /// Flag to remember if the SCoP contained an error block or not.
  bool HasErrorBlock;

  /// Max loop depth.
  unsigned MaxLoopDepth;

  /// Number of copy statements.
  unsigned CopyStmtsNum;

  typedef std::list<ScopStmt> StmtSet;
  /// The statements in this Scop.
  StmtSet Stmts;

  /// Parameters of this Scop
  ParameterSetTy Parameters;

  /// Mapping from parameters to their ids.
  DenseMap<const SCEV *, isl_id *> ParameterIds;

  /// The context of the SCoP created during SCoP detection.
  ScopDetection::DetectionContext &DC;

  /// Isl context.
  ///
  /// We need a shared_ptr with reference counter to delete the context when all
  /// isl objects are deleted. We will distribute the shared_ptr to all objects
  /// that use the context to create isl objects, and increase the reference
  /// counter. By doing this, we guarantee that the context is deleted when we
  /// delete the last object that creates isl objects with the context.
  std::shared_ptr<isl_ctx> IslCtx;

  /// A map from basic blocks to SCoP statements.
  DenseMap<BasicBlock *, ScopStmt *> StmtMap;

  /// A map from basic blocks to their domains.
  DenseMap<BasicBlock *, isl_set *> DomainMap;

  /// Constraints on parameters.
  isl_set *Context;

  /// The affinator used to translate SCEVs to isl expressions.
  SCEVAffinator Affinator;

  typedef std::map<std::pair<AssertingVH<const Value>, int>,
                   std::unique_ptr<ScopArrayInfo>>
      ArrayInfoMapTy;

  typedef StringMap<std::unique_ptr<ScopArrayInfo>> ArrayNameMapTy;

  typedef SetVector<ScopArrayInfo *> ArrayInfoSetTy;

  /// A map to remember ScopArrayInfo objects for all base pointers.
  ///
  /// As PHI nodes may have two array info objects associated, we add a flag
  /// that distinguishes between the PHI node specific ArrayInfo object
  /// and the normal one.
  ArrayInfoMapTy ScopArrayInfoMap;

  /// A map to remember ScopArrayInfo objects for all names of memory
  ///        references.
  ArrayNameMapTy ScopArrayNameMap;

  /// A set to remember ScopArrayInfo objects.
  /// @see Scop::ScopArrayInfoMap
  ArrayInfoSetTy ScopArrayInfoSet;

  /// The assumptions under which this scop was built.
  ///
  /// When constructing a scop sometimes the exact representation of a statement
  /// or condition would be very complex, but there is a common case which is a
  /// lot simpler, but which is only valid under certain assumptions. The
  /// assumed context records the assumptions taken during the construction of
  /// this scop and that need to be code generated as a run-time test.
  isl_set *AssumedContext;

  /// The restrictions under which this SCoP was built.
  ///
  /// The invalid context is similar to the assumed context as it contains
  /// constraints over the parameters. However, while we need the constraints
  /// in the assumed context to be "true" the constraints in the invalid context
  /// need to be "false". Otherwise they behave the same.
  isl_set *InvalidContext;

  /// Helper struct to remember assumptions.
  struct Assumption {

    /// The kind of the assumption (e.g., WRAPPING).
    AssumptionKind Kind;

    /// Flag to distinguish assumptions and restrictions.
    AssumptionSign Sign;

    /// The valid/invalid context if this is an assumption/restriction.
    isl_set *Set;

    /// The location that caused this assumption.
    DebugLoc Loc;

    /// An optional block whose domain can simplify the assumption.
    BasicBlock *BB;
  };

  /// Collection to hold taken assumptions.
  ///
  /// There are two reasons why we want to record assumptions first before we
  /// add them to the assumed/invalid context:
  ///   1) If the SCoP is not profitable or otherwise invalid without the
  ///      assumed/invalid context we do not have to compute it.
  ///   2) Information about the context are gathered rather late in the SCoP
  ///      construction (basically after we know all parameters), thus the user
  ///      might see overly complicated assumptions to be taken while they will
  ///      only be simplified later on.
  SmallVector<Assumption, 8> RecordedAssumptions;

  /// The schedule of the SCoP
  ///
  /// The schedule of the SCoP describes the execution order of the statements
  /// in the scop by assigning each statement instance a possibly
  /// multi-dimensional execution time. The schedule is stored as a tree of
  /// schedule nodes.
  ///
  /// The most common nodes in a schedule tree are so-called band nodes. Band
  /// nodes map statement instances into a multi dimensional schedule space.
  /// This space can be seen as a multi-dimensional clock.
  ///
  /// Example:
  ///
  /// <S,(5,4)>  may be mapped to (5,4) by this schedule:
  ///
  /// s0 = i (Year of execution)
  /// s1 = j (Day of execution)
  ///
  /// or to (9, 20) by this schedule:
  ///
  /// s0 = i + j (Year of execution)
  /// s1 = 20 (Day of execution)
  ///
  /// The order statement instances are executed is defined by the
  /// schedule vectors they are mapped to. A statement instance
  /// <A, (i, j, ..)> is executed before a statement instance <B, (i', ..)>, if
  /// the schedule vector of A is lexicographic smaller than the schedule
  /// vector of B.
  ///
  /// Besides band nodes, schedule trees contain additional nodes that specify
  /// a textual ordering between two subtrees or filter nodes that filter the
  /// set of statement instances that will be scheduled in a subtree. There
  /// are also several other nodes. A full description of the different nodes
  /// in a schedule tree is given in the isl manual.
  isl_schedule *Schedule;

  /// The set of minimal/maximal accesses for each alias group.
  ///
  /// When building runtime alias checks we look at all memory instructions and
  /// build so called alias groups. Each group contains a set of accesses to
  /// different base arrays which might alias with each other. However, between
  /// alias groups there is no aliasing possible.
  ///
  /// In a program with int and float pointers annotated with tbaa information
  /// we would probably generate two alias groups, one for the int pointers and
  /// one for the float pointers.
  ///
  /// During code generation we will create a runtime alias check for each alias
  /// group to ensure the SCoP is executed in an alias free environment.
  MinMaxVectorPairVectorTy MinMaxAliasGroups;

  /// Mapping from invariant loads to the representing invariant load of
  ///        their equivalence class.
  ValueToValueMap InvEquivClassVMap;

  /// List of invariant accesses.
  InvariantEquivClassesTy InvariantEquivClasses;

  /// Scop constructor; invoked from ScopBuilder::buildScop.
  Scop(Region &R, ScalarEvolution &SE, LoopInfo &LI,
       ScopDetection::DetectionContext &DC);

  //@}

  /// Initialize this ScopBuilder.
  void init(AliasAnalysis &AA, AssumptionCache &AC, DominatorTree &DT,
            LoopInfo &LI);

  /// Propagate domains that are known due to graph properties.
  ///
  /// As a CFG is mostly structured we use the graph properties to propagate
  /// domains without the need to compute all path conditions. In particular, if
  /// a block A dominates a block B and B post-dominates A we know that the
  /// domain of B is a superset of the domain of A. As we do not have
  /// post-dominator information available here we use the less precise region
  /// information. Given a region R, we know that the exit is always executed if
  /// the entry was executed, thus the domain of the exit is a superset of the
  /// domain of the entry. In case the exit can only be reached from within the
  /// region the domains are in fact equal. This function will use this property
  /// to avoid the generation of condition constraints that determine when a
  /// branch is taken. If @p BB is a region entry block we will propagate its
  /// domain to the region exit block. Additionally, we put the region exit
  /// block in the @p FinishedExitBlocks set so we can later skip edges from
  /// within the region to that block.
  ///
  /// @param BB The block for which the domain is currently propagated.
  /// @param BBLoop The innermost affine loop surrounding @p BB.
  /// @param FinishedExitBlocks Set of region exits the domain was set for.
  /// @param LI The LoopInfo for the current function.
  ///
  void propagateDomainConstraintsToRegionExit(
      BasicBlock *BB, Loop *BBLoop,
      SmallPtrSetImpl<BasicBlock *> &FinishedExitBlocks, LoopInfo &LI);

  /// Compute the union of predecessor domains for @p BB.
  ///
  /// To compute the union of all domains of predecessors of @p BB this
  /// function applies similar reasoning on the CFG structure as described for
  ///   @see propagateDomainConstraintsToRegionExit
  ///
  /// @param BB     The block for which the predecessor domains are collected.
  /// @param Domain The domain under which BB is executed.
  /// @param DT     The DominatorTree for the current function.
  /// @param LI     The LoopInfo for the current function.
  ///
  /// @returns The domain under which @p BB is executed.
  __isl_give isl_set *
  getPredecessorDomainConstraints(BasicBlock *BB, __isl_keep isl_set *Domain,
                                  DominatorTree &DT, LoopInfo &LI);

  /// Add loop carried constraints to the header block of the loop @p L.
  ///
  /// @param L  The loop to process.
  /// @param LI The LoopInfo for the current function.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool addLoopBoundsToHeaderDomain(Loop *L, LoopInfo &LI);

  /// Compute the branching constraints for each basic block in @p R.
  ///
  /// @param R  The region we currently build branching conditions for.
  /// @param DT The DominatorTree for the current function.
  /// @param LI The LoopInfo for the current function.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool buildDomainsWithBranchConstraints(Region *R, DominatorTree &DT,
                                         LoopInfo &LI);

  /// Propagate the domain constraints through the region @p R.
  ///
  /// @param R  The region we currently build branching conditions for.
  /// @param DT The DominatorTree for the current function.
  /// @param LI The LoopInfo for the current function.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool propagateDomainConstraints(Region *R, DominatorTree &DT, LoopInfo &LI);

  /// Propagate invalid domains of statements through @p R.
  ///
  /// This method will propagate invalid statement domains through @p R and at
  /// the same time add error block domains to them. Additionally, the domains
  /// of error statements and those only reachable via error statements will be
  /// replaced by an empty set. Later those will be removed completely.
  ///
  /// @param R  The currently traversed region.
  /// @param DT The DominatorTree for the current function.
  /// @param LI The LoopInfo for the current function.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool propagateInvalidStmtDomains(Region *R, DominatorTree &DT, LoopInfo &LI);

  /// Compute the domain for each basic block in @p R.
  ///
  /// @param R  The region we currently traverse.
  /// @param DT The DominatorTree for the current function.
  /// @param LI The LoopInfo for the current function.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool buildDomains(Region *R, DominatorTree &DT, LoopInfo &LI);

  /// Add parameter constraints to @p C that imply a non-empty domain.
  __isl_give isl_set *addNonEmptyDomainConstraints(__isl_take isl_set *C) const;

  /// Simplify the SCoP representation
  void simplifySCoP(bool AfterHoisting);

  /// Return the access for the base ptr of @p MA if any.
  MemoryAccess *lookupBasePtrAccess(MemoryAccess *MA);

  /// Check if the base ptr of @p MA is in the SCoP but not hoistable.
  bool hasNonHoistableBasePtrInScop(MemoryAccess *MA,
                                    __isl_keep isl_union_map *Writes);

  /// Create equivalence classes for required invariant accesses.
  ///
  /// These classes will consolidate multiple required invariant loads from the
  /// same address in order to keep the number of dimensions in the SCoP
  /// description small. For each such class equivalence class only one
  /// representing element, hence one required invariant load, will be chosen
  /// and modeled as parameter. The method
  /// Scop::getRepresentingInvariantLoadSCEV() will replace each element from an
  /// equivalence class with the representing element that is modeled. As a
  /// consequence Scop::getIdForParam() will only return an id for the
  /// representing element of each equivalence class, thus for each required
  /// invariant location.
  void buildInvariantEquivalenceClasses();

  /// Return the context under which the access cannot be hoisted.
  ///
  /// @param Access The access to check.
  /// @param Writes The set of all memory writes in the scop.
  ///
  /// @return Return the context under which the access cannot be hoisted or a
  ///         nullptr if it cannot be hoisted at all.
  __isl_give isl_set *getNonHoistableCtx(MemoryAccess *Access,
                                         __isl_keep isl_union_map *Writes);

  /// Verify that all required invariant loads have been hoisted.
  ///
  /// Invariant load hoisting is not guaranteed to hoist all loads that were
  /// assumed to be scop invariant during scop detection. This function checks
  /// for cases where the hoisting failed, but where it would have been
  /// necessary for our scop modeling to be correct. In case of insufficent
  /// hoisting the scop is marked as invalid.
  ///
  /// In the example below Bound[1] is required to be invariant:
  ///
  /// for (int i = 1; i < Bound[0]; i++)
  ///   for (int j = 1; j < Bound[1]; j++)
  ///     ...
  ///
  void verifyInvariantLoads();

  /// Hoist invariant memory loads and check for required ones.
  ///
  /// We first identify "common" invariant loads, thus loads that are invariant
  /// and can be hoisted. Then we check if all required invariant loads have
  /// been identified as (common) invariant. A load is a required invariant load
  /// if it was assumed to be invariant during SCoP detection, e.g., to assume
  /// loop bounds to be affine or runtime alias checks to be placeable. In case
  /// a required invariant load was not identified as (common) invariant we will
  /// drop this SCoP. An example for both "common" as well as required invariant
  /// loads is given below:
  ///
  /// for (int i = 1; i < *LB[0]; i++)
  ///   for (int j = 1; j < *LB[1]; j++)
  ///     A[i][j] += A[0][0] + (*V);
  ///
  /// Common inv. loads: V, A[0][0], LB[0], LB[1]
  /// Required inv. loads: LB[0], LB[1], (V, if it may alias with A or LB)
  ///
  void hoistInvariantLoads();

  /// Add invariant loads listed in @p InvMAs with the domain of @p Stmt.
  void addInvariantLoads(ScopStmt &Stmt, InvariantAccessesTy &InvMAs);

  /// Create an id for @p Param and store it in the ParameterIds map.
  void createParameterId(const SCEV *Param);

  /// Build the Context of the Scop.
  void buildContext();

  /// Add user provided parameter constraints to context (source code).
  void addUserAssumptions(AssumptionCache &AC, DominatorTree &DT, LoopInfo &LI);

  /// Add user provided parameter constraints to context (command line).
  void addUserContext();

  /// Add the bounds of the parameters to the context.
  void addParameterBounds();

  /// Simplify the assumed and invalid context.
  void simplifyContexts();

  /// Get the representing SCEV for @p S if applicable, otherwise @p S.
  ///
  /// Invariant loads of the same location are put in an equivalence class and
  /// only one of them is chosen as a representing element that will be
  /// modeled as a parameter. The others have to be normalized, i.e.,
  /// replaced by the representing element of their equivalence class, in order
  /// to get the correct parameter value, e.g., in the SCEVAffinator.
  ///
  /// @param S The SCEV to normalize.
  ///
  /// @return The representing SCEV for invariant loads or @p S if none.
  const SCEV *getRepresentingInvariantLoadSCEV(const SCEV *S);

  /// Create a new SCoP statement for either @p BB or @p R.
  ///
  /// Either @p BB or @p R should be non-null. A new statement for the non-null
  /// argument will be created and added to the statement vector and map.
  ///
  /// @param BB         The basic block we build the statement for (or null)
  /// @param R          The region we build the statement for (or null).
  void addScopStmt(BasicBlock *BB, Region *R);

  /// @param Update access dimensionalities.
  ///
  /// When detecting memory accesses different accesses to the same array may
  /// have built with different dimensionality, as outer zero-values dimensions
  /// may not have been recognized as separate dimensions. This function goes
  /// again over all memory accesses and updates their dimensionality to match
  /// the dimensionality of the underlying ScopArrayInfo object.
  void updateAccessDimensionality();

  /// Construct the schedule of this SCoP.
  ///
  /// @param LI The LoopInfo for the current function.
  void buildSchedule(LoopInfo &LI);

  /// A loop stack element to keep track of per-loop information during
  ///        schedule construction.
  typedef struct LoopStackElement {
    // The loop for which we keep information.
    Loop *L;

    // The (possibly incomplete) schedule for this loop.
    isl_schedule *Schedule;

    // The number of basic blocks in the current loop, for which a schedule has
    // already been constructed.
    unsigned NumBlocksProcessed;

    LoopStackElement(Loop *L, __isl_give isl_schedule *S,
                     unsigned NumBlocksProcessed)
        : L(L), Schedule(S), NumBlocksProcessed(NumBlocksProcessed) {}
  } LoopStackElementTy;

  /// The loop stack used for schedule construction.
  ///
  /// The loop stack keeps track of schedule information for a set of nested
  /// loops as well as an (optional) 'nullptr' loop that models the outermost
  /// schedule dimension. The loops in a loop stack always have a parent-child
  /// relation where the loop at position n is the parent of the loop at
  /// position n + 1.
  typedef SmallVector<LoopStackElementTy, 4> LoopStackTy;

  /// Construct schedule information for a given Region and add the
  ///        derived information to @p LoopStack.
  ///
  /// Given a Region we derive schedule information for all RegionNodes
  /// contained in this region ensuring that the assigned execution times
  /// correctly model the existing control flow relations.
  ///
  /// @param R              The region which to process.
  /// @param LoopStack      A stack of loops that are currently under
  ///                       construction.
  /// @param LI The LoopInfo for the current function.
  void buildSchedule(Region *R, LoopStackTy &LoopStack, LoopInfo &LI);

  /// Build Schedule for the region node @p RN and add the derived
  ///        information to @p LoopStack.
  ///
  /// In case @p RN is a BasicBlock or a non-affine Region, we construct the
  /// schedule for this @p RN and also finalize loop schedules in case the
  /// current @p RN completes the loop.
  ///
  /// In case @p RN is a not-non-affine Region, we delegate the construction to
  /// buildSchedule(Region *R, ...).
  ///
  /// @param RN             The RegionNode region traversed.
  /// @param LoopStack      A stack of loops that are currently under
  ///                       construction.
  /// @param LI The LoopInfo for the current function.
  void buildSchedule(RegionNode *RN, LoopStackTy &LoopStack, LoopInfo &LI);

  /// Collect all memory access relations of a given type.
  ///
  /// @param Predicate A predicate function that returns true if an access is
  ///                  of a given type.
  ///
  /// @returns The set of memory accesses in the scop that match the predicate.
  __isl_give isl_union_map *
  getAccessesOfType(std::function<bool(MemoryAccess &)> Predicate);

  /// @name Helper functions for printing the Scop.
  ///
  //@{
  void printContext(raw_ostream &OS) const;
  void printArrayInfo(raw_ostream &OS) const;
  void printStatements(raw_ostream &OS) const;
  void printAliasAssumptions(raw_ostream &OS) const;
  //@}

  friend class ScopBuilder;

public:
  ~Scop();

  /// Get the count of copy statements added to this Scop.
  ///
  /// @return The count of copy statements added to this Scop.
  unsigned getCopyStmtsNum() { return CopyStmtsNum; }

  /// Create a new copy statement.
  ///
  /// A new statement will be created and added to the statement vector.
  ///
  /// @param Stmt       The parent statement.
  /// @param SourceRel  The source location.
  /// @param TargetRel  The target location.
  /// @param Domain     The original domain under which copy statement whould
  ///                   be executed.
  ScopStmt *addScopStmt(__isl_take isl_map *SourceRel,
                        __isl_take isl_map *TargetRel,
                        __isl_take isl_set *Domain);

  /// Add the access function to all MemoryAccess objects of the Scop
  ///        created in this pass.
  void addAccessFunction(MemoryAccess *Access) {
    AccessFunctions.emplace_back(Access);
  }

  ScalarEvolution *getSE() const;

  /// Get the count of parameters used in this Scop.
  ///
  /// @return The count of parameters used in this Scop.
  size_t getNumParams() const { return Parameters.size(); }

  /// Take a list of parameters and add the new ones to the scop.
  void addParams(const ParameterSetTy &NewParameters);

  /// Return whether this scop is empty, i.e. contains no statements that
  /// could be executed.
  bool isEmpty() const { return Stmts.empty(); }

  typedef ArrayInfoSetTy::iterator array_iterator;
  typedef ArrayInfoSetTy::const_iterator const_array_iterator;
  typedef iterator_range<ArrayInfoSetTy::iterator> array_range;
  typedef iterator_range<ArrayInfoSetTy::const_iterator> const_array_range;

  inline array_iterator array_begin() { return ScopArrayInfoSet.begin(); }

  inline array_iterator array_end() { return ScopArrayInfoSet.end(); }

  inline const_array_iterator array_begin() const {
    return ScopArrayInfoSet.begin();
  }

  inline const_array_iterator array_end() const {
    return ScopArrayInfoSet.end();
  }

  inline array_range arrays() {
    return array_range(array_begin(), array_end());
  }

  inline const_array_range arrays() const {
    return const_array_range(array_begin(), array_end());
  }

  /// Return the isl_id that represents a certain parameter.
  ///
  /// @param Parameter A SCEV that was recognized as a Parameter.
  ///
  /// @return The corresponding isl_id or NULL otherwise.
  __isl_give isl_id *getIdForParam(const SCEV *Parameter);

  /// Get the maximum region of this static control part.
  ///
  /// @return The maximum region of this static control part.
  inline const Region &getRegion() const { return R; }
  inline Region &getRegion() { return R; }

  /// Return the function this SCoP is in.
  Function &getFunction() const { return *R.getEntry()->getParent(); }

  /// Check if @p L is contained in the SCoP.
  bool contains(const Loop *L) const { return R.contains(L); }

  /// Check if @p BB is contained in the SCoP.
  bool contains(const BasicBlock *BB) const { return R.contains(BB); }

  /// Check if @p I is contained in the SCoP.
  bool contains(const Instruction *I) const { return R.contains(I); }

  /// Return the unique exit block of the SCoP.
  BasicBlock *getExit() const { return R.getExit(); }

  /// Return the unique exiting block of the SCoP if any.
  BasicBlock *getExitingBlock() const { return R.getExitingBlock(); }

  /// Return the unique entry block of the SCoP.
  BasicBlock *getEntry() const { return R.getEntry(); }

  /// Return the unique entering block of the SCoP if any.
  BasicBlock *getEnteringBlock() const { return R.getEnteringBlock(); }

  /// Return true if @p BB is the exit block of the SCoP.
  bool isExit(BasicBlock *BB) const { return getExit() == BB; }

  /// Return a range of all basic blocks in the SCoP.
  Region::block_range blocks() const { return R.blocks(); }

  /// Return true if and only if @p BB dominates the SCoP.
  bool isDominatedBy(const DominatorTree &DT, BasicBlock *BB) const;

  /// Get the maximum depth of the loop.
  ///
  /// @return The maximum depth of the loop.
  inline unsigned getMaxLoopDepth() const { return MaxLoopDepth; }

  /// Return the invariant equivalence class for @p Val if any.
  InvariantEquivClassTy *lookupInvariantEquivClass(Value *Val);

  /// Return the set of invariant accesses.
  InvariantEquivClassesTy &getInvariantAccesses() {
    return InvariantEquivClasses;
  }

  /// Check if the scop has any invariant access.
  bool hasInvariantAccesses() { return !InvariantEquivClasses.empty(); }

  /// Mark the SCoP as optimized by the scheduler.
  void markAsOptimized() { IsOptimized = true; }

  /// Check if the SCoP has been optimized by the scheduler.
  bool isOptimized() const { return IsOptimized; }

  /// Get the name of this Scop.
  std::string getNameStr() const;

  /// Get the constraint on parameter of this Scop.
  ///
  /// @return The constraint on parameter of this Scop.
  __isl_give isl_set *getContext() const;
  __isl_give isl_space *getParamSpace() const;

  /// Get the assumed context for this Scop.
  ///
  /// @return The assumed context of this Scop.
  __isl_give isl_set *getAssumedContext() const;

  /// Return true if the optimized SCoP can be executed.
  ///
  /// In addition to the runtime check context this will also utilize the domain
  /// constraints to decide it the optimized version can actually be executed.
  ///
  /// @returns True if the optimized SCoP can be executed.
  bool hasFeasibleRuntimeContext() const;

  /// Check if the assumption in @p Set is trivial or not.
  ///
  /// @param Set  The relations between parameters that are assumed to hold.
  /// @param Sign Enum to indicate if the assumptions in @p Set are positive
  ///             (needed/assumptions) or negative (invalid/restrictions).
  ///
  /// @returns True if the assumption @p Set is not trivial.
  bool isEffectiveAssumption(__isl_keep isl_set *Set, AssumptionSign Sign);

  /// Track and report an assumption.
  ///
  /// Use 'clang -Rpass-analysis=polly-scops' or 'opt
  /// -pass-remarks-analysis=polly-scops' to output the assumptions.
  ///
  /// @param Kind The assumption kind describing the underlying cause.
  /// @param Set  The relations between parameters that are assumed to hold.
  /// @param Loc  The location in the source that caused this assumption.
  /// @param Sign Enum to indicate if the assumptions in @p Set are positive
  ///             (needed/assumptions) or negative (invalid/restrictions).
  ///
  /// @returns True if the assumption is not trivial.
  bool trackAssumption(AssumptionKind Kind, __isl_keep isl_set *Set,
                       DebugLoc Loc, AssumptionSign Sign);

  /// Add assumptions to assumed context.
  ///
  /// The assumptions added will be assumed to hold during the execution of the
  /// scop. However, as they are generally not statically provable, at code
  /// generation time run-time checks will be generated that ensure the
  /// assumptions hold.
  ///
  /// WARNING: We currently exploit in simplifyAssumedContext the knowledge
  ///          that assumptions do not change the set of statement instances
  ///          executed.
  ///
  /// @param Kind The assumption kind describing the underlying cause.
  /// @param Set  The relations between parameters that are assumed to hold.
  /// @param Loc  The location in the source that caused this assumption.
  /// @param Sign Enum to indicate if the assumptions in @p Set are positive
  ///             (needed/assumptions) or negative (invalid/restrictions).
  void addAssumption(AssumptionKind Kind, __isl_take isl_set *Set, DebugLoc Loc,
                     AssumptionSign Sign);

  /// Record an assumption for later addition to the assumed context.
  ///
  /// This function will add the assumption to the RecordedAssumptions. This
  /// collection will be added (@see addAssumption) to the assumed context once
  /// all paramaters are known and the context is fully build.
  ///
  /// @param Kind The assumption kind describing the underlying cause.
  /// @param Set  The relations between parameters that are assumed to hold.
  /// @param Loc  The location in the source that caused this assumption.
  /// @param Sign Enum to indicate if the assumptions in @p Set are positive
  ///             (needed/assumptions) or negative (invalid/restrictions).
  /// @param BB   The block in which this assumption was taken. If it is
  ///             set, the domain of that block will be used to simplify the
  ///             actual assumption in @p Set once it is added. This is useful
  ///             if the assumption was created prior to the domain.
  void recordAssumption(AssumptionKind Kind, __isl_take isl_set *Set,
                        DebugLoc Loc, AssumptionSign Sign,
                        BasicBlock *BB = nullptr);

  /// Add all recorded assumptions to the assumed context.
  void addRecordedAssumptions();

  /// Mark the scop as invalid.
  ///
  /// This method adds an assumption to the scop that is always invalid. As a
  /// result, the scop will not be optimized later on. This function is commonly
  /// called when a condition makes it impossible (or too compile time
  /// expensive) to process this scop any further.
  ///
  /// @param Kind The assumption kind describing the underlying cause.
  /// @param Loc  The location in the source that triggered .
  void invalidate(AssumptionKind Kind, DebugLoc Loc);

  /// Get the invalid context for this Scop.
  ///
  /// @return The invalid context of this Scop.
  __isl_give isl_set *getInvalidContext() const;

  /// Return true if and only if the InvalidContext is trivial (=empty).
  bool hasTrivialInvalidContext() const {
    return isl_set_is_empty(InvalidContext);
  }

  /// Build the alias checks for this SCoP.
  bool buildAliasChecks(AliasAnalysis &AA);

  /// Build all alias groups for this SCoP.
  ///
  /// @returns True if __no__ error occurred, false otherwise.
  bool buildAliasGroups(AliasAnalysis &AA);

  /// Return all alias groups for this SCoP.
  const MinMaxVectorPairVectorTy &getAliasGroups() const {
    return MinMaxAliasGroups;
  }

  /// Get an isl string representing the context.
  std::string getContextStr() const;

  /// Get an isl string representing the assumed context.
  std::string getAssumedContextStr() const;

  /// Get an isl string representing the invalid context.
  std::string getInvalidContextStr() const;

  /// Return the ScopStmt for the given @p BB or nullptr if there is
  ///        none.
  ScopStmt *getStmtFor(BasicBlock *BB) const;

  /// Return the ScopStmt that represents the Region @p R, or nullptr if
  ///        it is not represented by any statement in this Scop.
  ScopStmt *getStmtFor(Region *R) const;

  /// Return the ScopStmt that represents @p RN; can return nullptr if
  ///        the RegionNode is not within the SCoP or has been removed due to
  ///        simplifications.
  ScopStmt *getStmtFor(RegionNode *RN) const;

  /// Return the ScopStmt an instruction belongs to, or nullptr if it
  ///        does not belong to any statement in this Scop.
  ScopStmt *getStmtFor(Instruction *Inst) const {
    return getStmtFor(Inst->getParent());
  }

  /// Return the number of statements in the SCoP.
  size_t getSize() const { return Stmts.size(); }

  /// @name Statements Iterators
  ///
  /// These iterators iterate over all statements of this Scop.
  //@{
  typedef StmtSet::iterator iterator;
  typedef StmtSet::const_iterator const_iterator;

  iterator begin() { return Stmts.begin(); }
  iterator end() { return Stmts.end(); }
  const_iterator begin() const { return Stmts.begin(); }
  const_iterator end() const { return Stmts.end(); }

  typedef StmtSet::reverse_iterator reverse_iterator;
  typedef StmtSet::const_reverse_iterator const_reverse_iterator;

  reverse_iterator rbegin() { return Stmts.rbegin(); }
  reverse_iterator rend() { return Stmts.rend(); }
  const_reverse_iterator rbegin() const { return Stmts.rbegin(); }
  const_reverse_iterator rend() const { return Stmts.rend(); }
  //@}

  /// Return the set of required invariant loads.
  const InvariantLoadsSetTy &getRequiredInvariantLoads() const {
    return DC.RequiredILS;
  }

  /// Add @p LI to the set of required invariant loads.
  void addRequiredInvariantLoad(LoadInst *LI) { DC.RequiredILS.insert(LI); }

  /// Return true if and only if @p LI is a required invariant load.
  bool isRequiredInvariantLoad(LoadInst *LI) const {
    return getRequiredInvariantLoads().count(LI);
  }

  /// Return the set of boxed (thus overapproximated) loops.
  const BoxedLoopsSetTy &getBoxedLoops() const { return DC.BoxedLoopsSet; }

  /// Return true if and only if @p R is a non-affine subregion.
  bool isNonAffineSubRegion(const Region *R) {
    return DC.NonAffineSubRegionSet.count(R);
  }

  const MapInsnToMemAcc &getInsnToMemAccMap() const { return DC.InsnToMemAcc; }

  /// Return the (possibly new) ScopArrayInfo object for @p Access.
  ///
  /// @param ElementType The type of the elements stored in this array.
  /// @param Kind        The kind of the array info object.
  /// @param BaseName    The optional name of this memory reference.
  const ScopArrayInfo *getOrCreateScopArrayInfo(Value *BasePtr,
                                                Type *ElementType,
                                                ArrayRef<const SCEV *> Sizes,
                                                ScopArrayInfo::MemoryKind Kind,
                                                const char *BaseName = nullptr);

  /// Create an array and return the corresponding ScopArrayInfo object.
  ///
  /// @param ElementType The type of the elements stored in this array.
  /// @param BaseName    The name of this memory reference.
  /// @param Sizes       The sizes of dimensions.
  const ScopArrayInfo *createScopArrayInfo(Type *ElementType,
                                           const std::string &BaseName,
                                           const std::vector<unsigned> &Sizes);

  /// Return the cached ScopArrayInfo object for @p BasePtr.
  ///
  /// @param BasePtr   The base pointer the object has been stored for.
  /// @param Kind      The kind of array info object.
  const ScopArrayInfo *getScopArrayInfo(Value *BasePtr,
                                        ScopArrayInfo::MemoryKind Kind);

  /// Invalidate ScopArrayInfo object for base address.
  ///
  /// @param BasePtr The base pointer of the ScopArrayInfo object to invalidate.
  /// @param Kind    The Kind of the ScopArrayInfo object.
  void invalidateScopArrayInfo(Value *BasePtr, ScopArrayInfo::MemoryKind Kind) {
    auto It = ScopArrayInfoMap.find(std::make_pair(BasePtr, Kind));
    if (It == ScopArrayInfoMap.end())
      return;
    ScopArrayInfoSet.remove(It->second.get());
    ScopArrayInfoMap.erase(It);
  }

  void setContext(__isl_take isl_set *NewContext);

  /// Align the parameters in the statement to the scop context
  void realignParams();

  /// Return true if this SCoP can be profitably optimized.
  bool isProfitable() const;

  /// Return true if the SCoP contained at least one error block.
  bool hasErrorBlock() const { return HasErrorBlock; }

  /// Return true if the underlying region has a single exiting block.
  bool hasSingleExitEdge() const { return HasSingleExitEdge; }

  /// Print the static control part.
  ///
  /// @param OS The output stream the static control part is printed to.
  void print(raw_ostream &OS) const;

  /// Print the ScopStmt to stderr.
  void dump() const;

  /// Get the isl context of this static control part.
  ///
  /// @return The isl context of this static control part.
  isl_ctx *getIslCtx() const;

  /// Directly return the shared_ptr of the context.
  const std::shared_ptr<isl_ctx> &getSharedIslCtx() const { return IslCtx; }

  /// Compute the isl representation for the SCEV @p E
  ///
  /// @param E  The SCEV that should be translated.
  /// @param BB An (optional) basic block in which the isl_pw_aff is computed.
  ///           SCEVs known to not reference any loops in the SCoP can be
  ///           passed without a @p BB.
  /// @param NonNegative Flag to indicate the @p E has to be non-negative.
  ///
  /// Note that this function will always return a valid isl_pw_aff. However, if
  /// the translation of @p E was deemed to complex the SCoP is invalidated and
  /// a dummy value of appropriate dimension is returned. This allows to bail
  /// for complex cases without "error handling code" needed on the users side.
  __isl_give PWACtx getPwAff(const SCEV *E, BasicBlock *BB = nullptr,
                             bool NonNegative = false);

  /// Compute the isl representation for the SCEV @p E
  ///
  /// This function is like @see Scop::getPwAff() but strips away the invalid
  /// domain part associated with the piecewise affine function.
  __isl_give isl_pw_aff *getPwAffOnly(const SCEV *E, BasicBlock *BB = nullptr);

  /// Return the domain of @p Stmt.
  ///
  /// @param Stmt The statement for which the conditions should be returned.
  __isl_give isl_set *getDomainConditions(const ScopStmt *Stmt) const;

  /// Return the domain of @p BB.
  ///
  /// @param BB The block for which the conditions should be returned.
  __isl_give isl_set *getDomainConditions(BasicBlock *BB) const;

  /// Get a union set containing the iteration domains of all statements.
  __isl_give isl_union_set *getDomains() const;

  /// Get a union map of all may-writes performed in the SCoP.
  __isl_give isl_union_map *getMayWrites();

  /// Get a union map of all must-writes performed in the SCoP.
  __isl_give isl_union_map *getMustWrites();

  /// Get a union map of all writes performed in the SCoP.
  __isl_give isl_union_map *getWrites();

  /// Get a union map of all reads performed in the SCoP.
  __isl_give isl_union_map *getReads();

  /// Get a union map of all memory accesses performed in the SCoP.
  __isl_give isl_union_map *getAccesses();

  /// Get the schedule of all the statements in the SCoP.
  ///
  /// @return The schedule of all the statements in the SCoP, if the schedule of
  /// the Scop does not contain extension nodes, and nullptr, otherwise.
  __isl_give isl_union_map *getSchedule() const;

  /// Get a schedule tree describing the schedule of all statements.
  __isl_give isl_schedule *getScheduleTree() const;

  /// Update the current schedule
  ///
  /// NewSchedule The new schedule (given as a flat union-map).
  void setSchedule(__isl_take isl_union_map *NewSchedule);

  /// Update the current schedule
  ///
  /// NewSchedule The new schedule (given as schedule tree).
  void setScheduleTree(__isl_take isl_schedule *NewSchedule);

  /// Intersects the domains of all statements in the SCoP.
  ///
  /// @return true if a change was made
  bool restrictDomains(__isl_take isl_union_set *Domain);

  /// Get the depth of a loop relative to the outermost loop in the Scop.
  ///
  /// This will return
  ///    0 if @p L is an outermost loop in the SCoP
  ///   >0 for other loops in the SCoP
  ///   -1 if @p L is nullptr or there is no outermost loop in the SCoP
  int getRelativeLoopDepth(const Loop *L) const;

  /// Find the ScopArrayInfo associated with an isl Id
  ///        that has name @p Name.
  ScopArrayInfo *getArrayInfoByName(const std::string BaseName);

  /// Check whether @p Schedule contains extension nodes.
  ///
  /// @return true if @p Schedule contains extension nodes.
  static bool containsExtensionNode(__isl_keep isl_schedule *Schedule);
};

/// Print Scop scop to raw_ostream O.
static inline raw_ostream &operator<<(raw_ostream &O, const Scop &scop) {
  scop.print(O);
  return O;
}

/// The legacy pass manager's analysis pass to compute scop information
///        for a region.
class ScopInfoRegionPass : public RegionPass {
  /// The Scop pointer which is used to construct a Scop.
  std::unique_ptr<Scop> S;

public:
  static char ID; // Pass identification, replacement for typeid

  ScopInfoRegionPass() : RegionPass(ID) {}
  ~ScopInfoRegionPass() {}

  /// Build Scop object, the Polly IR of static control
  ///        part for the current SESE-Region.
  ///
  /// @return If the current region is a valid for a static control part,
  ///         return the Polly IR representing this static control part,
  ///         return null otherwise.
  Scop *getScop() { return S.get(); }
  const Scop *getScop() const { return S.get(); }

  /// Calculate the polyhedral scop information for a given Region.
  bool runOnRegion(Region *R, RGPassManager &RGM) override;

  void releaseMemory() override { S.reset(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

//===----------------------------------------------------------------------===//
/// The legacy pass manager's analysis pass to compute scop information
///        for the whole function.
///
/// This pass will maintain a map of the maximal region within a scop to its
/// scop object for all the feasible scops present in a function.
/// This pass is an alternative to the ScopInfoRegionPass in order to avoid a
/// region pass manager.
class ScopInfoWrapperPass : public FunctionPass {

public:
  using RegionToScopMapTy = DenseMap<Region *, std::unique_ptr<Scop>>;
  using iterator = RegionToScopMapTy::iterator;
  using const_iterator = RegionToScopMapTy::const_iterator;

private:
  /// A map of Region to its Scop object containing
  ///        Polly IR of static control part
  RegionToScopMapTy RegionToScopMap;

public:
  static char ID; // Pass identification, replacement for typeid

  ScopInfoWrapperPass() : FunctionPass(ID) {}
  ~ScopInfoWrapperPass() {}

  /// Get the Scop object for the given Region
  ///
  /// @return If the given region is the maximal region within a scop, return
  ///         the scop object. If the given region is a subregion, return a
  ///         nullptr. Top level region containing the entry block of a function
  ///         is not considered in the scop creation.
  Scop *getScop(Region *R) const {
    auto MapIt = RegionToScopMap.find(R);
    if (MapIt != RegionToScopMap.end())
      return MapIt->second.get();
    return nullptr;
  }

  iterator begin() { return RegionToScopMap.begin(); }
  iterator end() { return RegionToScopMap.end(); }
  const_iterator begin() const { return RegionToScopMap.begin(); }
  const_iterator end() const { return RegionToScopMap.end(); }

  /// Calculate all the polyhedral scops for a given function.
  bool runOnFunction(Function &F) override;

  void releaseMemory() override { RegionToScopMap.clear(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // end namespace polly

namespace llvm {
class PassRegistry;
void initializeScopInfoRegionPassPass(llvm::PassRegistry &);
void initializeScopInfoWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif
