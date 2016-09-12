//===------ PPCGCodeGeneration.cpp - Polly Accelerator Code Generation. ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Take a scop created by ScopInfo and map it to GPU code using the ppcg
// GPU mapping strategy.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/CodeGen/Utils.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/Support/SCEVValidator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "isl/union_map.h"

extern "C" {
#include "ppcg/cuda.h"
#include "ppcg/gpu.h"
#include "ppcg/gpu_print.h"
#include "ppcg/ppcg.h"
#include "ppcg/schedule.h"
}

#include "llvm/Support/Debug.h"

using namespace polly;
using namespace llvm;

#define DEBUG_TYPE "polly-codegen-ppcg"

static cl::opt<bool> DumpSchedule("polly-acc-dump-schedule",
                                  cl::desc("Dump the computed GPU Schedule"),
                                  cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                  cl::cat(PollyCategory));

static cl::opt<bool>
    DumpCode("polly-acc-dump-code",
             cl::desc("Dump C code describing the GPU mapping"), cl::Hidden,
             cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> DumpKernelIR("polly-acc-dump-kernel-ir",
                                  cl::desc("Dump the kernel LLVM-IR"),
                                  cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                  cl::cat(PollyCategory));

static cl::opt<bool> DumpKernelASM("polly-acc-dump-kernel-asm",
                                   cl::desc("Dump the kernel assembly code"),
                                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                   cl::cat(PollyCategory));

static cl::opt<bool> FastMath("polly-acc-fastmath",
                              cl::desc("Allow unsafe math optimizations"),
                              cl::Hidden, cl::init(false), cl::ZeroOrMore,
                              cl::cat(PollyCategory));
static cl::opt<bool> SharedMemory("polly-acc-use-shared",
                                  cl::desc("Use shared memory"), cl::Hidden,
                                  cl::init(false), cl::ZeroOrMore,
                                  cl::cat(PollyCategory));
static cl::opt<bool> PrivateMemory("polly-acc-use-private",
                                   cl::desc("Use private memory"), cl::Hidden,
                                   cl::init(false), cl::ZeroOrMore,
                                   cl::cat(PollyCategory));

static cl::opt<std::string>
    CudaVersion("polly-acc-cuda-version",
                cl::desc("The CUDA version to compile for"), cl::Hidden,
                cl::init("sm_30"), cl::ZeroOrMore, cl::cat(PollyCategory));

/// Create the ast expressions for a ScopStmt.
///
/// This function is a callback for to generate the ast expressions for each
/// of the scheduled ScopStmts.
static __isl_give isl_id_to_ast_expr *pollyBuildAstExprForStmt(
    void *StmtT, isl_ast_build *Build,
    isl_multi_pw_aff *(*FunctionIndex)(__isl_take isl_multi_pw_aff *MPA,
                                       isl_id *Id, void *User),
    void *UserIndex,
    isl_ast_expr *(*FunctionExpr)(isl_ast_expr *Expr, isl_id *Id, void *User),
    void *UserExpr) {

  ScopStmt *Stmt = (ScopStmt *)StmtT;

  isl_ctx *Ctx;

  if (!Stmt || !Build)
    return NULL;

  Ctx = isl_ast_build_get_ctx(Build);
  isl_id_to_ast_expr *RefToExpr = isl_id_to_ast_expr_alloc(Ctx, 0);

  for (MemoryAccess *Acc : *Stmt) {
    isl_map *AddrFunc = Acc->getAddressFunction();
    AddrFunc = isl_map_intersect_domain(AddrFunc, Stmt->getDomain());
    isl_id *RefId = Acc->getId();
    isl_pw_multi_aff *PMA = isl_pw_multi_aff_from_map(AddrFunc);
    isl_multi_pw_aff *MPA = isl_multi_pw_aff_from_pw_multi_aff(PMA);
    MPA = isl_multi_pw_aff_coalesce(MPA);
    MPA = FunctionIndex(MPA, RefId, UserIndex);
    isl_ast_expr *Access = isl_ast_build_access_from_multi_pw_aff(Build, MPA);
    Access = FunctionExpr(Access, RefId, UserExpr);
    RefToExpr = isl_id_to_ast_expr_set(RefToExpr, RefId, Access);
  }

  return RefToExpr;
}

/// Generate code for a GPU specific isl AST.
///
/// The GPUNodeBuilder augments the general existing IslNodeBuilder, which
/// generates code for general-prupose AST nodes, with special functionality
/// for generating GPU specific user nodes.
///
/// @see GPUNodeBuilder::createUser
class GPUNodeBuilder : public IslNodeBuilder {
public:
  GPUNodeBuilder(PollyIRBuilder &Builder, ScopAnnotator &Annotator, Pass *P,
                 const DataLayout &DL, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, Scop &S, gpu_prog *Prog)
      : IslNodeBuilder(Builder, Annotator, P, DL, LI, SE, DT, S), Prog(Prog) {
    getExprBuilder().setIDToSAI(&IDToSAI);
  }

  /// Create after-run-time-check initialization code.
  void initializeAfterRTH();

  /// Finalize the generated scop.
  virtual void finalize();

  /// Track if the full build process was successful.
  ///
  /// This value is set to false, if throughout the build process an error
  /// occurred which prevents us from generating valid GPU code.
  bool BuildSuccessful = true;

private:
  /// A vector of array base pointers for which a new ScopArrayInfo was created.
  ///
  /// This vector is used to delete the ScopArrayInfo when it is not needed any
  /// more.
  std::vector<Value *> LocalArrays;

  /// A map from ScopArrays to their corresponding device allocations.
  std::map<ScopArrayInfo *, Value *> DeviceAllocations;

  /// The current GPU context.
  Value *GPUContext;

  /// The set of isl_ids allocated in the kernel
  std::vector<isl_id *> KernelIds;

  /// A module containing GPU code.
  ///
  /// This pointer is only set in case we are currently generating GPU code.
  std::unique_ptr<Module> GPUModule;

  /// The GPU program we generate code for.
  gpu_prog *Prog;

  /// Class to free isl_ids.
  class IslIdDeleter {
  public:
    void operator()(__isl_take isl_id *Id) { isl_id_free(Id); };
  };

  /// A set containing all isl_ids allocated in a GPU kernel.
  ///
  /// By releasing this set all isl_ids will be freed.
  std::set<std::unique_ptr<isl_id, IslIdDeleter>> KernelIDs;

  IslExprBuilder::IDToScopArrayInfoTy IDToSAI;

  /// Create code for user-defined AST nodes.
  ///
  /// These AST nodes can be of type:
  ///
  ///   - ScopStmt:      A computational statement (TODO)
  ///   - Kernel:        A GPU kernel call (TODO)
  ///   - Data-Transfer: A GPU <-> CPU data-transfer
  ///   - In-kernel synchronization
  ///   - In-kernel memory copy statement
  ///
  /// @param UserStmt The ast node to generate code for.
  virtual void createUser(__isl_take isl_ast_node *UserStmt);

  enum DataDirection { HOST_TO_DEVICE, DEVICE_TO_HOST };

  /// Create code for a data transfer statement
  ///
  /// @param TransferStmt The data transfer statement.
  /// @param Direction The direction in which to transfer data.
  void createDataTransfer(__isl_take isl_ast_node *TransferStmt,
                          enum DataDirection Direction);

  /// Find llvm::Values referenced in GPU kernel.
  ///
  /// @param Kernel The kernel to scan for llvm::Values
  ///
  /// @returns A set of values referenced by the kernel.
  SetVector<Value *> getReferencesInKernel(ppcg_kernel *Kernel);

  /// Compute the sizes of the execution grid for a given kernel.
  ///
  /// @param Kernel The kernel to compute grid sizes for.
  ///
  /// @returns A tuple with grid sizes for X and Y dimension
  std::tuple<Value *, Value *> getGridSizes(ppcg_kernel *Kernel);

  /// Compute the sizes of the thread blocks for a given kernel.
  ///
  /// @param Kernel The kernel to compute thread block sizes for.
  ///
  /// @returns A tuple with thread block sizes for X, Y, and Z dimensions.
  std::tuple<Value *, Value *, Value *> getBlockSizes(ppcg_kernel *Kernel);

  /// Create kernel launch parameters.
  ///
  /// @param Kernel        The kernel to create parameters for.
  /// @param F             The kernel function that has been created.
  /// @param SubtreeValues The set of llvm::Values referenced by this kernel.
  ///
  /// @returns A stack allocated array with pointers to the parameter
  ///          values that are passed to the kernel.
  Value *createLaunchParameters(ppcg_kernel *Kernel, Function *F,
                                SetVector<Value *> SubtreeValues);

  /// Create declarations for kernel variable.
  ///
  /// This includes shared memory declarations.
  ///
  /// @param Kernel        The kernel definition to create variables for.
  /// @param FN            The function into which to generate the variables.
  void createKernelVariables(ppcg_kernel *Kernel, Function *FN);

  /// Add CUDA annotations to module.
  ///
  /// Add a set of CUDA annotations that declares the maximal block dimensions
  /// that will be used to execute the CUDA kernel. This allows the NVIDIA
  /// PTX compiler to bound the number of allocated registers to ensure the
  /// resulting kernel is known to run with up to as many block dimensions
  /// as specified here.
  ///
  /// @param M         The module to add the annotations to.
  /// @param BlockDimX The size of block dimension X.
  /// @param BlockDimY The size of block dimension Y.
  /// @param BlockDimZ The size of block dimension Z.
  void addCUDAAnnotations(Module *M, Value *BlockDimX, Value *BlockDimY,
                          Value *BlockDimZ);

  /// Create GPU kernel.
  ///
  /// Code generate the kernel described by @p KernelStmt.
  ///
  /// @param KernelStmt The ast node to generate kernel code for.
  void createKernel(__isl_take isl_ast_node *KernelStmt);

  /// Generate code that computes the size of an array.
  ///
  /// @param Array The array for which to compute a size.
  Value *getArraySize(gpu_array_info *Array);

  /// Prepare the kernel arguments for kernel code generation
  ///
  /// @param Kernel The kernel to generate code for.
  /// @param FN     The function created for the kernel.
  void prepareKernelArguments(ppcg_kernel *Kernel, Function *FN);

  /// Create kernel function.
  ///
  /// Create a kernel function located in a newly created module that can serve
  /// as target for device code generation. Set the Builder to point to the
  /// start block of this newly created function.
  ///
  /// @param Kernel The kernel to generate code for.
  /// @param SubtreeValues The set of llvm::Values referenced by this kernel.
  void createKernelFunction(ppcg_kernel *Kernel,
                            SetVector<Value *> &SubtreeValues);

  /// Create the declaration of a kernel function.
  ///
  /// The kernel function takes as arguments:
  ///
  ///   - One i8 pointer for each external array reference used in the kernel.
  ///   - Host iterators
  ///   - Parameters
  ///   - Other LLVM Value references (TODO)
  ///
  /// @param Kernel The kernel to generate the function declaration for.
  /// @param SubtreeValues The set of llvm::Values referenced by this kernel.
  ///
  /// @returns The newly declared function.
  Function *createKernelFunctionDecl(ppcg_kernel *Kernel,
                                     SetVector<Value *> &SubtreeValues);

  /// Insert intrinsic functions to obtain thread and block ids.
  ///
  /// @param The kernel to generate the intrinsic functions for.
  void insertKernelIntrinsics(ppcg_kernel *Kernel);

  /// Create a global-to-shared or shared-to-global copy statement.
  ///
  /// @param CopyStmt The copy statement to generate code for
  void createKernelCopy(ppcg_kernel_stmt *CopyStmt);

  /// Create code for a ScopStmt called in @p Expr.
  ///
  /// @param Expr The expression containing the call.
  /// @param KernelStmt The kernel statement referenced in the call.
  void createScopStmt(isl_ast_expr *Expr, ppcg_kernel_stmt *KernelStmt);

  /// Create an in-kernel synchronization call.
  void createKernelSync();

  /// Create a PTX assembly string for the current GPU kernel.
  ///
  /// @returns A string containing the corresponding PTX assembly code.
  std::string createKernelASM();

  /// Remove references from the dominator tree to the kernel function @p F.
  ///
  /// @param F The function to remove references to.
  void clearDominators(Function *F);

  /// Remove references from scalar evolution to the kernel function @p F.
  ///
  /// @param F The function to remove references to.
  void clearScalarEvolution(Function *F);

  /// Remove references from loop info to the kernel function @p F.
  ///
  /// @param F The function to remove references to.
  void clearLoops(Function *F);

  /// Finalize the generation of the kernel function.
  ///
  /// Free the LLVM-IR module corresponding to the kernel and -- if requested --
  /// dump its IR to stderr.
  ///
  /// @returns The Assembly string of the kernel.
  std::string finalizeKernelFunction();

  /// Create code that allocates memory to store arrays on device.
  void allocateDeviceArrays();

  /// Free all allocated device arrays.
  void freeDeviceArrays();

  /// Create a call to initialize the GPU context.
  ///
  /// @returns A pointer to the newly initialized context.
  Value *createCallInitContext();

  /// Create a call to get the device pointer for a kernel allocation.
  ///
  /// @param Allocation The Polly GPU allocation
  ///
  /// @returns The device parameter corresponding to this allocation.
  Value *createCallGetDevicePtr(Value *Allocation);

  /// Create a call to free the GPU context.
  ///
  /// @param Context A pointer to an initialized GPU context.
  void createCallFreeContext(Value *Context);

  /// Create a call to allocate memory on the device.
  ///
  /// @param Size The size of memory to allocate
  ///
  /// @returns A pointer that identifies this allocation.
  Value *createCallAllocateMemoryForDevice(Value *Size);

  /// Create a call to free a device array.
  ///
  /// @param Array The device array to free.
  void createCallFreeDeviceMemory(Value *Array);

  /// Create a call to copy data from host to device.
  ///
  /// @param HostPtr A pointer to the host data that should be copied.
  /// @param DevicePtr A device pointer specifying the location to copy to.
  void createCallCopyFromHostToDevice(Value *HostPtr, Value *DevicePtr,
                                      Value *Size);

  /// Create a call to copy data from device to host.
  ///
  /// @param DevicePtr A pointer to the device data that should be copied.
  /// @param HostPtr A host pointer specifying the location to copy to.
  void createCallCopyFromDeviceToHost(Value *DevicePtr, Value *HostPtr,
                                      Value *Size);

  /// Create a call to get a kernel from an assembly string.
  ///
  /// @param Buffer The string describing the kernel.
  /// @param Entry  The name of the kernel function to call.
  ///
  /// @returns A pointer to a kernel object
  Value *createCallGetKernel(Value *Buffer, Value *Entry);

  /// Create a call to free a GPU kernel.
  ///
  /// @param GPUKernel THe kernel to free.
  void createCallFreeKernel(Value *GPUKernel);

  /// Create a call to launch a GPU kernel.
  ///
  /// @param GPUKernel  The kernel to launch.
  /// @param GridDimX   The size of the first grid dimension.
  /// @param GridDimY   The size of the second grid dimension.
  /// @param GridBlockX The size of the first block dimension.
  /// @param GridBlockY The size of the second block dimension.
  /// @param GridBlockZ The size of the third block dimension.
  /// @param Paramters  A pointer to an array that contains itself pointers to
  ///                   the parameter values passed for each kernel argument.
  void createCallLaunchKernel(Value *GPUKernel, Value *GridDimX,
                              Value *GridDimY, Value *BlockDimX,
                              Value *BlockDimY, Value *BlockDimZ,
                              Value *Parameters);
};

void GPUNodeBuilder::initializeAfterRTH() {
  BasicBlock *NewBB = SplitBlock(Builder.GetInsertBlock(),
                                 &*Builder.GetInsertPoint(), &DT, &LI);
  NewBB->setName("polly.acc.initialize");
  Builder.SetInsertPoint(&NewBB->front());

  GPUContext = createCallInitContext();
  allocateDeviceArrays();
}

void GPUNodeBuilder::finalize() {
  freeDeviceArrays();
  createCallFreeContext(GPUContext);
  IslNodeBuilder::finalize();
}

void GPUNodeBuilder::allocateDeviceArrays() {
  isl_ast_build *Build = isl_ast_build_from_context(S.getContext());

  for (int i = 0; i < Prog->n_array; ++i) {
    gpu_array_info *Array = &Prog->array[i];
    auto *ScopArray = (ScopArrayInfo *)Array->user;
    std::string DevArrayName("p_dev_array_");
    DevArrayName.append(Array->name);

    Value *ArraySize = getArraySize(Array);
    Value *DevArray = createCallAllocateMemoryForDevice(ArraySize);
    DevArray->setName(DevArrayName);
    DeviceAllocations[ScopArray] = DevArray;
  }

  isl_ast_build_free(Build);
}

void GPUNodeBuilder::addCUDAAnnotations(Module *M, Value *BlockDimX,
                                        Value *BlockDimY, Value *BlockDimZ) {
  auto AnnotationNode = M->getOrInsertNamedMetadata("nvvm.annotations");

  for (auto &F : *M) {
    if (F.getCallingConv() != CallingConv::PTX_Kernel)
      continue;

    Value *V[] = {BlockDimX, BlockDimY, BlockDimZ};

    Metadata *Elements[] = {
        ValueAsMetadata::get(&F),   MDString::get(M->getContext(), "maxntidx"),
        ValueAsMetadata::get(V[0]), MDString::get(M->getContext(), "maxntidy"),
        ValueAsMetadata::get(V[1]), MDString::get(M->getContext(), "maxntidz"),
        ValueAsMetadata::get(V[2]),
    };
    MDNode *Node = MDNode::get(M->getContext(), Elements);
    AnnotationNode->addOperand(Node);
  }
}

void GPUNodeBuilder::freeDeviceArrays() {
  for (auto &Array : DeviceAllocations)
    createCallFreeDeviceMemory(Array.second);
}

Value *GPUNodeBuilder::createCallGetKernel(Value *Buffer, Value *Entry) {
  const char *Name = "polly_getKernel";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {Buffer, Entry});
}

Value *GPUNodeBuilder::createCallGetDevicePtr(Value *Allocation) {
  const char *Name = "polly_getDevicePtr";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {Allocation});
}

void GPUNodeBuilder::createCallLaunchKernel(Value *GPUKernel, Value *GridDimX,
                                            Value *GridDimY, Value *BlockDimX,
                                            Value *BlockDimY, Value *BlockDimZ,
                                            Value *Parameters) {
  const char *Name = "polly_launchKernel";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {GPUKernel, GridDimX, GridDimY, BlockDimX, BlockDimY,
                         BlockDimZ, Parameters});
}

void GPUNodeBuilder::createCallFreeKernel(Value *GPUKernel) {
  const char *Name = "polly_freeKernel";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {GPUKernel});
}

void GPUNodeBuilder::createCallFreeDeviceMemory(Value *Array) {
  const char *Name = "polly_freeDeviceMemory";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {Array});
}

Value *GPUNodeBuilder::createCallAllocateMemoryForDevice(Value *Size) {
  const char *Name = "polly_allocateMemoryForDevice";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt64Ty());
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {Size});
}

void GPUNodeBuilder::createCallCopyFromHostToDevice(Value *HostData,
                                                    Value *DeviceData,
                                                    Value *Size) {
  const char *Name = "polly_copyFromHostToDevice";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt64Ty());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {HostData, DeviceData, Size});
}

void GPUNodeBuilder::createCallCopyFromDeviceToHost(Value *DeviceData,
                                                    Value *HostData,
                                                    Value *Size) {
  const char *Name = "polly_copyFromDeviceToHost";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt64Ty());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {DeviceData, HostData, Size});
}

Value *GPUNodeBuilder::createCallInitContext() {
  const char *Name = "polly_initContext";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {});
}

void GPUNodeBuilder::createCallFreeContext(Value *Context) {
  const char *Name = "polly_freeContext";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {Context});
}

/// Check if one string is a prefix of another.
///
/// @param String The string in which to look for the prefix.
/// @param Prefix The prefix to look for.
static bool isPrefix(std::string String, std::string Prefix) {
  return String.find(Prefix) == 0;
}

Value *GPUNodeBuilder::getArraySize(gpu_array_info *Array) {
  isl_ast_build *Build = isl_ast_build_from_context(S.getContext());
  Value *ArraySize = ConstantInt::get(Builder.getInt64Ty(), Array->size);

  if (!gpu_array_is_scalar(Array)) {
    auto OffsetDimZero = isl_pw_aff_copy(Array->bound[0]);
    isl_ast_expr *Res = isl_ast_build_expr_from_pw_aff(Build, OffsetDimZero);

    for (unsigned int i = 1; i < Array->n_index; i++) {
      isl_pw_aff *Bound_I = isl_pw_aff_copy(Array->bound[i]);
      isl_ast_expr *Expr = isl_ast_build_expr_from_pw_aff(Build, Bound_I);
      Res = isl_ast_expr_mul(Res, Expr);
    }

    Value *NumElements = ExprBuilder.create(Res);
    ArraySize = Builder.CreateMul(ArraySize, NumElements);
  }
  isl_ast_build_free(Build);
  return ArraySize;
}

void GPUNodeBuilder::createDataTransfer(__isl_take isl_ast_node *TransferStmt,
                                        enum DataDirection Direction) {
  isl_ast_expr *Expr = isl_ast_node_user_get_expr(TransferStmt);
  isl_ast_expr *Arg = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(Arg);
  auto Array = (gpu_array_info *)isl_id_get_user(Id);
  auto ScopArray = (ScopArrayInfo *)(Array->user);

  Value *Size = getArraySize(Array);
  Value *DevPtr = DeviceAllocations[ScopArray];

  Value *HostPtr;

  if (gpu_array_is_scalar(Array))
    HostPtr = BlockGen.getOrCreateAlloca(ScopArray);
  else
    HostPtr = ScopArray->getBasePtr();

  HostPtr = Builder.CreatePointerCast(HostPtr, Builder.getInt8PtrTy());

  if (Direction == HOST_TO_DEVICE)
    createCallCopyFromHostToDevice(HostPtr, DevPtr, Size);
  else
    createCallCopyFromDeviceToHost(DevPtr, HostPtr, Size);

  isl_id_free(Id);
  isl_ast_expr_free(Arg);
  isl_ast_expr_free(Expr);
  isl_ast_node_free(TransferStmt);
}

void GPUNodeBuilder::createUser(__isl_take isl_ast_node *UserStmt) {
  isl_ast_expr *Expr = isl_ast_node_user_get_expr(UserStmt);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(StmtExpr);
  isl_id_free(Id);
  isl_ast_expr_free(StmtExpr);

  const char *Str = isl_id_get_name(Id);
  if (!strcmp(Str, "kernel")) {
    createKernel(UserStmt);
    isl_ast_expr_free(Expr);
    return;
  }

  if (isPrefix(Str, "to_device")) {
    createDataTransfer(UserStmt, HOST_TO_DEVICE);
    isl_ast_expr_free(Expr);
    return;
  }

  if (isPrefix(Str, "from_device")) {
    createDataTransfer(UserStmt, DEVICE_TO_HOST);
    isl_ast_expr_free(Expr);
    return;
  }

  isl_id *Anno = isl_ast_node_get_annotation(UserStmt);
  struct ppcg_kernel_stmt *KernelStmt =
      (struct ppcg_kernel_stmt *)isl_id_get_user(Anno);
  isl_id_free(Anno);

  switch (KernelStmt->type) {
  case ppcg_kernel_domain:
    createScopStmt(Expr, KernelStmt);
    isl_ast_node_free(UserStmt);
    return;
  case ppcg_kernel_copy:
    createKernelCopy(KernelStmt);
    isl_ast_expr_free(Expr);
    isl_ast_node_free(UserStmt);
    return;
  case ppcg_kernel_sync:
    createKernelSync();
    isl_ast_expr_free(Expr);
    isl_ast_node_free(UserStmt);
    return;
  }

  isl_ast_expr_free(Expr);
  isl_ast_node_free(UserStmt);
  return;
}
void GPUNodeBuilder::createKernelCopy(ppcg_kernel_stmt *KernelStmt) {
  isl_ast_expr *LocalIndex = isl_ast_expr_copy(KernelStmt->u.c.local_index);
  LocalIndex = isl_ast_expr_address_of(LocalIndex);
  Value *LocalAddr = ExprBuilder.create(LocalIndex);
  isl_ast_expr *Index = isl_ast_expr_copy(KernelStmt->u.c.index);
  Index = isl_ast_expr_address_of(Index);
  Value *GlobalAddr = ExprBuilder.create(Index);

  if (KernelStmt->u.c.read) {
    LoadInst *Load = Builder.CreateLoad(GlobalAddr, "shared.read");
    Builder.CreateStore(Load, LocalAddr);
  } else {
    LoadInst *Load = Builder.CreateLoad(LocalAddr, "shared.write");
    Builder.CreateStore(Load, GlobalAddr);
  }
}

void GPUNodeBuilder::createScopStmt(isl_ast_expr *Expr,
                                    ppcg_kernel_stmt *KernelStmt) {
  auto Stmt = (ScopStmt *)KernelStmt->u.d.stmt->stmt;
  isl_id_to_ast_expr *Indexes = KernelStmt->u.d.ref2expr;

  LoopToScevMapT LTS;
  LTS.insert(OutsideLoopIterations.begin(), OutsideLoopIterations.end());

  createSubstitutions(Expr, Stmt, LTS);

  if (Stmt->isBlockStmt())
    BlockGen.copyStmt(*Stmt, LTS, Indexes);
  else
    assert(0 && "Region statement not supported\n");
}

void GPUNodeBuilder::createKernelSync() {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  auto *Sync = Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0);
  Builder.CreateCall(Sync, {});
}

/// Collect llvm::Values referenced from @p Node
///
/// This function only applies to isl_ast_nodes that are user_nodes referring
/// to a ScopStmt. All other node types are ignore.
///
/// @param Node The node to collect references for.
/// @param User A user pointer used as storage for the data that is collected.
///
/// @returns isl_bool_true if data could be collected successfully.
isl_bool collectReferencesInGPUStmt(__isl_keep isl_ast_node *Node, void *User) {
  if (isl_ast_node_get_type(Node) != isl_ast_node_user)
    return isl_bool_true;

  isl_ast_expr *Expr = isl_ast_node_user_get_expr(Node);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(StmtExpr);
  const char *Str = isl_id_get_name(Id);
  isl_id_free(Id);
  isl_ast_expr_free(StmtExpr);
  isl_ast_expr_free(Expr);

  if (!isPrefix(Str, "Stmt"))
    return isl_bool_true;

  Id = isl_ast_node_get_annotation(Node);
  auto *KernelStmt = (ppcg_kernel_stmt *)isl_id_get_user(Id);
  auto Stmt = (ScopStmt *)KernelStmt->u.d.stmt->stmt;
  isl_id_free(Id);

  addReferencesFromStmt(Stmt, User, false /* CreateScalarRefs */);

  return isl_bool_true;
}

SetVector<Value *> GPUNodeBuilder::getReferencesInKernel(ppcg_kernel *Kernel) {
  SetVector<Value *> SubtreeValues;
  SetVector<const SCEV *> SCEVs;
  SetVector<const Loop *> Loops;
  SubtreeReferences References = {
      LI, SE, S, ValueMap, SubtreeValues, SCEVs, getBlockGenerator()};

  for (const auto &I : IDToValue)
    SubtreeValues.insert(I.second);

  isl_ast_node_foreach_descendant_top_down(
      Kernel->tree, collectReferencesInGPUStmt, &References);

  for (const SCEV *Expr : SCEVs)
    findValues(Expr, SE, SubtreeValues);

  for (auto &SAI : S.arrays())
    SubtreeValues.remove(SAI->getBasePtr());

  isl_space *Space = S.getParamSpace();
  for (long i = 0; i < isl_space_dim(Space, isl_dim_param); i++) {
    isl_id *Id = isl_space_get_dim_id(Space, isl_dim_param, i);
    assert(IDToValue.count(Id));
    Value *Val = IDToValue[Id];
    SubtreeValues.remove(Val);
    isl_id_free(Id);
  }
  isl_space_free(Space);

  for (long i = 0; i < isl_space_dim(Kernel->space, isl_dim_set); i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
    assert(IDToValue.count(Id));
    Value *Val = IDToValue[Id];
    SubtreeValues.remove(Val);
    isl_id_free(Id);
  }

  return SubtreeValues;
}

void GPUNodeBuilder::clearDominators(Function *F) {
  DomTreeNode *N = DT.getNode(&F->getEntryBlock());
  std::vector<BasicBlock *> Nodes;
  for (po_iterator<DomTreeNode *> I = po_begin(N), E = po_end(N); I != E; ++I)
    Nodes.push_back(I->getBlock());

  for (BasicBlock *BB : Nodes)
    DT.eraseNode(BB);
}

void GPUNodeBuilder::clearScalarEvolution(Function *F) {
  for (BasicBlock &BB : *F) {
    Loop *L = LI.getLoopFor(&BB);
    if (L)
      SE.forgetLoop(L);
  }
}

void GPUNodeBuilder::clearLoops(Function *F) {
  for (BasicBlock &BB : *F) {
    Loop *L = LI.getLoopFor(&BB);
    if (L)
      SE.forgetLoop(L);
    LI.removeBlock(&BB);
  }
}

std::tuple<Value *, Value *> GPUNodeBuilder::getGridSizes(ppcg_kernel *Kernel) {
  std::vector<Value *> Sizes;
  isl_ast_build *Context = isl_ast_build_from_context(S.getContext());

  for (long i = 0; i < Kernel->n_grid; i++) {
    isl_pw_aff *Size = isl_multi_pw_aff_get_pw_aff(Kernel->grid_size, i);
    isl_ast_expr *GridSize = isl_ast_build_expr_from_pw_aff(Context, Size);
    Value *Res = ExprBuilder.create(GridSize);
    Res = Builder.CreateTrunc(Res, Builder.getInt32Ty());
    Sizes.push_back(Res);
  }
  isl_ast_build_free(Context);

  for (long i = Kernel->n_grid; i < 3; i++)
    Sizes.push_back(ConstantInt::get(Builder.getInt32Ty(), 1));

  return std::make_tuple(Sizes[0], Sizes[1]);
}

std::tuple<Value *, Value *, Value *>
GPUNodeBuilder::getBlockSizes(ppcg_kernel *Kernel) {
  std::vector<Value *> Sizes;

  for (long i = 0; i < Kernel->n_block; i++) {
    Value *Res = ConstantInt::get(Builder.getInt32Ty(), Kernel->block_dim[i]);
    Sizes.push_back(Res);
  }

  for (long i = Kernel->n_block; i < 3; i++)
    Sizes.push_back(ConstantInt::get(Builder.getInt32Ty(), 1));

  return std::make_tuple(Sizes[0], Sizes[1], Sizes[2]);
}

Value *
GPUNodeBuilder::createLaunchParameters(ppcg_kernel *Kernel, Function *F,
                                       SetVector<Value *> SubtreeValues) {
  Type *ArrayTy = ArrayType::get(Builder.getInt8PtrTy(),
                                 std::distance(F->arg_begin(), F->arg_end()));

  BasicBlock *EntryBlock =
      &Builder.GetInsertBlock()->getParent()->getEntryBlock();
  std::string Launch = "polly_launch_" + std::to_string(Kernel->id);
  Instruction *Parameters =
      new AllocaInst(ArrayTy, Launch + "_params", EntryBlock->getTerminator());

  int Index = 0;
  for (long i = 0; i < Prog->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(Id);

    Value *DevArray = DeviceAllocations[(ScopArrayInfo *)SAI];
    DevArray = createCallGetDevicePtr(DevArray);
    Instruction *Param = new AllocaInst(
        Builder.getInt8PtrTy(), Launch + "_param_" + std::to_string(Index),
        EntryBlock->getTerminator());
    Builder.CreateStore(DevArray, Param);
    Value *Slot = Builder.CreateGEP(
        Parameters, {Builder.getInt64(0), Builder.getInt64(Index)});
    Value *ParamTyped =
        Builder.CreatePointerCast(Param, Builder.getInt8PtrTy());
    Builder.CreateStore(ParamTyped, Slot);
    Index++;
  }

  int NumHostIters = isl_space_dim(Kernel->space, isl_dim_set);

  for (long i = 0; i < NumHostIters; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
    Value *Val = IDToValue[Id];
    isl_id_free(Id);
    Instruction *Param = new AllocaInst(
        Val->getType(), Launch + "_param_" + std::to_string(Index),
        EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    Value *Slot = Builder.CreateGEP(
        Parameters, {Builder.getInt64(0), Builder.getInt64(Index)});
    Value *ParamTyped =
        Builder.CreatePointerCast(Param, Builder.getInt8PtrTy());
    Builder.CreateStore(ParamTyped, Slot);
    Index++;
  }

  int NumVars = isl_space_dim(Kernel->space, isl_dim_param);

  for (long i = 0; i < NumVars; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_param, i);
    Value *Val = IDToValue[Id];
    isl_id_free(Id);
    Instruction *Param = new AllocaInst(
        Val->getType(), Launch + "_param_" + std::to_string(Index),
        EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    Value *Slot = Builder.CreateGEP(
        Parameters, {Builder.getInt64(0), Builder.getInt64(Index)});
    Value *ParamTyped =
        Builder.CreatePointerCast(Param, Builder.getInt8PtrTy());
    Builder.CreateStore(ParamTyped, Slot);
    Index++;
  }

  for (auto Val : SubtreeValues) {
    Instruction *Param = new AllocaInst(
        Val->getType(), Launch + "_param_" + std::to_string(Index),
        EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    Value *Slot = Builder.CreateGEP(
        Parameters, {Builder.getInt64(0), Builder.getInt64(Index)});
    Value *ParamTyped =
        Builder.CreatePointerCast(Param, Builder.getInt8PtrTy());
    Builder.CreateStore(ParamTyped, Slot);
    Index++;
  }

  auto Location = EntryBlock->getTerminator();
  return new BitCastInst(Parameters, Builder.getInt8PtrTy(),
                         Launch + "_params_i8ptr", Location);
}

void GPUNodeBuilder::createKernel(__isl_take isl_ast_node *KernelStmt) {
  isl_id *Id = isl_ast_node_get_annotation(KernelStmt);
  ppcg_kernel *Kernel = (ppcg_kernel *)isl_id_get_user(Id);
  isl_id_free(Id);
  isl_ast_node_free(KernelStmt);

  Value *BlockDimX, *BlockDimY, *BlockDimZ;
  std::tie(BlockDimX, BlockDimY, BlockDimZ) = getBlockSizes(Kernel);

  SetVector<Value *> SubtreeValues = getReferencesInKernel(Kernel);

  assert(Kernel->tree && "Device AST of kernel node is empty");

  Instruction &HostInsertPoint = *Builder.GetInsertPoint();
  IslExprBuilder::IDToValueTy HostIDs = IDToValue;
  ValueMapT HostValueMap = ValueMap;
  BlockGenerator::ScalarAllocaMapTy HostScalarMap = ScalarMap;
  BlockGenerator::ScalarAllocaMapTy HostPHIOpMap = PHIOpMap;
  ScalarMap.clear();
  PHIOpMap.clear();

  SetVector<const Loop *> Loops;

  // Create for all loops we depend on values that contain the current loop
  // iteration. These values are necessary to generate code for SCEVs that
  // depend on such loops. As a result we need to pass them to the subfunction.
  for (const Loop *L : Loops) {
    const SCEV *OuterLIV = SE.getAddRecExpr(SE.getUnknown(Builder.getInt64(0)),
                                            SE.getUnknown(Builder.getInt64(1)),
                                            L, SCEV::FlagAnyWrap);
    Value *V = generateSCEV(OuterLIV);
    OutsideLoopIterations[L] = SE.getUnknown(V);
    SubtreeValues.insert(V);
  }

  createKernelFunction(Kernel, SubtreeValues);

  create(isl_ast_node_copy(Kernel->tree));

  Function *F = Builder.GetInsertBlock()->getParent();
  addCUDAAnnotations(F->getParent(), BlockDimX, BlockDimY, BlockDimZ);
  clearDominators(F);
  clearScalarEvolution(F);
  clearLoops(F);

  Builder.SetInsertPoint(&HostInsertPoint);
  IDToValue = HostIDs;

  ValueMap = std::move(HostValueMap);
  ScalarMap = std::move(HostScalarMap);
  PHIOpMap = std::move(HostPHIOpMap);
  EscapeMap.clear();
  IDToSAI.clear();
  Annotator.resetAlternativeAliasBases();
  for (auto &BasePtr : LocalArrays)
    S.invalidateScopArrayInfo(BasePtr, ScopArrayInfo::MK_Array);
  LocalArrays.clear();

  Value *Parameters = createLaunchParameters(Kernel, F, SubtreeValues);

  std::string ASMString = finalizeKernelFunction();
  std::string Name = "kernel_" + std::to_string(Kernel->id);
  Value *KernelString = Builder.CreateGlobalStringPtr(ASMString, Name);
  Value *NameString = Builder.CreateGlobalStringPtr(Name, Name + "_name");
  Value *GPUKernel = createCallGetKernel(KernelString, NameString);

  Value *GridDimX, *GridDimY;
  std::tie(GridDimX, GridDimY) = getGridSizes(Kernel);

  createCallLaunchKernel(GPUKernel, GridDimX, GridDimY, BlockDimX, BlockDimY,
                         BlockDimZ, Parameters);
  createCallFreeKernel(GPUKernel);

  for (auto Id : KernelIds)
    isl_id_free(Id);

  KernelIds.clear();
}

/// Compute the DataLayout string for the NVPTX backend.
///
/// @param is64Bit Are we looking for a 64 bit architecture?
static std::string computeNVPTXDataLayout(bool is64Bit) {
  std::string Ret = "e";

  if (!is64Bit)
    Ret += "-p:32:32";

  Ret += "-i64:64-v16:16-v32:32-n16:32:64";

  return Ret;
}

Function *
GPUNodeBuilder::createKernelFunctionDecl(ppcg_kernel *Kernel,
                                         SetVector<Value *> &SubtreeValues) {
  std::vector<Type *> Args;
  std::string Identifier = "kernel_" + std::to_string(Kernel->id);

  for (long i = 0; i < Prog->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    Args.push_back(Builder.getInt8PtrTy());
  }

  int NumHostIters = isl_space_dim(Kernel->space, isl_dim_set);

  for (long i = 0; i < NumHostIters; i++)
    Args.push_back(Builder.getInt64Ty());

  int NumVars = isl_space_dim(Kernel->space, isl_dim_param);

  for (long i = 0; i < NumVars; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_param, i);
    Value *Val = IDToValue[Id];
    isl_id_free(Id);
    Args.push_back(Val->getType());
  }

  for (auto *V : SubtreeValues)
    Args.push_back(V->getType());

  auto *FT = FunctionType::get(Builder.getVoidTy(), Args, false);
  auto *FN = Function::Create(FT, Function::ExternalLinkage, Identifier,
                              GPUModule.get());
  FN->setCallingConv(CallingConv::PTX_Kernel);

  auto Arg = FN->arg_begin();
  for (long i = 0; i < Kernel->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    Arg->setName(Kernel->array[i].array->name);

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(isl_id_copy(Id));
    Type *EleTy = SAI->getElementType();
    Value *Val = &*Arg;
    SmallVector<const SCEV *, 4> Sizes;
    isl_ast_build *Build =
        isl_ast_build_from_context(isl_set_copy(Prog->context));
    for (long j = 1; j < Kernel->array[i].array->n_index; j++) {
      isl_ast_expr *DimSize = isl_ast_build_expr_from_pw_aff(
          Build, isl_pw_aff_copy(Kernel->array[i].array->bound[j]));
      auto V = ExprBuilder.create(DimSize);
      Sizes.push_back(SE.getSCEV(V));
    }
    const ScopArrayInfo *SAIRep =
        S.getOrCreateScopArrayInfo(Val, EleTy, Sizes, ScopArrayInfo::MK_Array);
    LocalArrays.push_back(Val);

    isl_ast_build_free(Build);
    KernelIds.push_back(Id);
    IDToSAI[Id] = SAIRep;
    Arg++;
  }

  for (long i = 0; i < NumHostIters; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
    Arg->setName(isl_id_get_name(Id));
    IDToValue[Id] = &*Arg;
    KernelIDs.insert(std::unique_ptr<isl_id, IslIdDeleter>(Id));
    Arg++;
  }

  for (long i = 0; i < NumVars; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_param, i);
    Arg->setName(isl_id_get_name(Id));
    Value *Val = IDToValue[Id];
    ValueMap[Val] = &*Arg;
    IDToValue[Id] = &*Arg;
    KernelIDs.insert(std::unique_ptr<isl_id, IslIdDeleter>(Id));
    Arg++;
  }

  for (auto *V : SubtreeValues) {
    Arg->setName(V->getName());
    ValueMap[V] = &*Arg;
    Arg++;
  }

  return FN;
}

void GPUNodeBuilder::insertKernelIntrinsics(ppcg_kernel *Kernel) {
  Intrinsic::ID IntrinsicsBID[] = {Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
                                   Intrinsic::nvvm_read_ptx_sreg_ctaid_y};

  Intrinsic::ID IntrinsicsTID[] = {Intrinsic::nvvm_read_ptx_sreg_tid_x,
                                   Intrinsic::nvvm_read_ptx_sreg_tid_y,
                                   Intrinsic::nvvm_read_ptx_sreg_tid_z};

  auto addId = [this](__isl_take isl_id *Id, Intrinsic::ID Intr) mutable {
    std::string Name = isl_id_get_name(Id);
    Module *M = Builder.GetInsertBlock()->getParent()->getParent();
    Function *IntrinsicFn = Intrinsic::getDeclaration(M, Intr);
    Value *Val = Builder.CreateCall(IntrinsicFn, {});
    Val = Builder.CreateIntCast(Val, Builder.getInt64Ty(), false, Name);
    IDToValue[Id] = Val;
    KernelIDs.insert(std::unique_ptr<isl_id, IslIdDeleter>(Id));
  };

  for (int i = 0; i < Kernel->n_grid; ++i) {
    isl_id *Id = isl_id_list_get_id(Kernel->block_ids, i);
    addId(Id, IntrinsicsBID[i]);
  }

  for (int i = 0; i < Kernel->n_block; ++i) {
    isl_id *Id = isl_id_list_get_id(Kernel->thread_ids, i);
    addId(Id, IntrinsicsTID[i]);
  }
}

void GPUNodeBuilder::prepareKernelArguments(ppcg_kernel *Kernel, Function *FN) {
  auto Arg = FN->arg_begin();
  for (long i = 0; i < Kernel->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(isl_id_copy(Id));
    isl_id_free(Id);

    if (SAI->getNumberOfDimensions() > 0) {
      Arg++;
      continue;
    }

    Value *Alloca = BlockGen.getOrCreateAlloca(SAI);
    Value *ArgPtr = &*Arg;
    Type *TypePtr = SAI->getElementType()->getPointerTo();
    Value *TypedArgPtr = Builder.CreatePointerCast(ArgPtr, TypePtr);
    Value *Val = Builder.CreateLoad(TypedArgPtr);
    Builder.CreateStore(Val, Alloca);

    Arg++;
  }
}

void GPUNodeBuilder::createKernelVariables(ppcg_kernel *Kernel, Function *FN) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();

  for (int i = 0; i < Kernel->n_var; ++i) {
    struct ppcg_kernel_var &Var = Kernel->var[i];
    isl_id *Id = isl_space_get_tuple_id(Var.array->space, isl_dim_set);
    Type *EleTy = ScopArrayInfo::getFromId(Id)->getElementType();

    Type *ArrayTy = EleTy;
    SmallVector<const SCEV *, 4> Sizes;

    for (unsigned int j = 1; j < Var.array->n_index; ++j) {
      isl_val *Val = isl_vec_get_element_val(Var.size, j);
      long Bound = isl_val_get_num_si(Val);
      isl_val_free(Val);
      Sizes.push_back(S.getSE()->getConstant(Builder.getInt64Ty(), Bound));
    }

    for (int j = Var.array->n_index - 1; j >= 0; --j) {
      isl_val *Val = isl_vec_get_element_val(Var.size, j);
      long Bound = isl_val_get_num_si(Val);
      isl_val_free(Val);
      ArrayTy = ArrayType::get(ArrayTy, Bound);
    }

    const ScopArrayInfo *SAI;
    Value *Allocation;
    if (Var.type == ppcg_access_shared) {
      auto GlobalVar = new GlobalVariable(
          *M, ArrayTy, false, GlobalValue::InternalLinkage, 0, Var.name,
          nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 3);
      GlobalVar->setAlignment(EleTy->getPrimitiveSizeInBits() / 8);
      GlobalVar->setInitializer(Constant::getNullValue(ArrayTy));

      Allocation = GlobalVar;
    } else if (Var.type == ppcg_access_private) {
      Allocation = Builder.CreateAlloca(ArrayTy, 0, "private_array");
    } else {
      llvm_unreachable("unknown variable type");
    }
    SAI = S.getOrCreateScopArrayInfo(Allocation, EleTy, Sizes,
                                     ScopArrayInfo::MK_Array);
    Id = isl_id_alloc(S.getIslCtx(), Var.name, nullptr);
    IDToValue[Id] = Allocation;
    LocalArrays.push_back(Allocation);
    KernelIds.push_back(Id);
    IDToSAI[Id] = SAI;
  }
}

void GPUNodeBuilder::createKernelFunction(ppcg_kernel *Kernel,
                                          SetVector<Value *> &SubtreeValues) {

  std::string Identifier = "kernel_" + std::to_string(Kernel->id);
  GPUModule.reset(new Module(Identifier, Builder.getContext()));
  GPUModule->setTargetTriple(Triple::normalize("nvptx64-nvidia-cuda"));
  GPUModule->setDataLayout(computeNVPTXDataLayout(true /* is64Bit */));

  Function *FN = createKernelFunctionDecl(Kernel, SubtreeValues);

  BasicBlock *PrevBlock = Builder.GetInsertBlock();
  auto EntryBlock = BasicBlock::Create(Builder.getContext(), "entry", FN);

  DominatorTree &DT = P->getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DT.addNewBlock(EntryBlock, PrevBlock);

  Builder.SetInsertPoint(EntryBlock);
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(EntryBlock, EntryBlock->begin());

  ScopDetection::markFunctionAsInvalid(FN);

  prepareKernelArguments(Kernel, FN);
  createKernelVariables(Kernel, FN);
  insertKernelIntrinsics(Kernel);
}

std::string GPUNodeBuilder::createKernelASM() {
  llvm::Triple GPUTriple(Triple::normalize("nvptx64-nvidia-cuda"));
  std::string ErrMsg;
  auto GPUTarget = TargetRegistry::lookupTarget(GPUTriple.getTriple(), ErrMsg);

  if (!GPUTarget) {
    errs() << ErrMsg << "\n";
    return "";
  }

  TargetOptions Options;
  Options.UnsafeFPMath = FastMath;
  std::unique_ptr<TargetMachine> TargetM(
      GPUTarget->createTargetMachine(GPUTriple.getTriple(), CudaVersion, "",
                                     Options, Optional<Reloc::Model>()));

  SmallString<0> ASMString;
  raw_svector_ostream ASMStream(ASMString);
  llvm::legacy::PassManager PM;

  PM.add(createTargetTransformInfoWrapperPass(TargetM->getTargetIRAnalysis()));

  if (TargetM->addPassesToEmitFile(
          PM, ASMStream, TargetMachine::CGFT_AssemblyFile, true /* verify */)) {
    errs() << "The target does not support generation of this file type!\n";
    return "";
  }

  PM.run(*GPUModule);

  return ASMStream.str();
}

std::string GPUNodeBuilder::finalizeKernelFunction() {
  if (verifyModule(*GPUModule)) {
    BuildSuccessful = false;
    return "";
  }

  if (DumpKernelIR)
    outs() << *GPUModule << "\n";

  // Optimize module.
  llvm::legacy::PassManager OptPasses;
  PassManagerBuilder PassBuilder;
  PassBuilder.OptLevel = 3;
  PassBuilder.SizeLevel = 0;
  PassBuilder.populateModulePassManager(OptPasses);
  OptPasses.run(*GPUModule);

  std::string Assembly = createKernelASM();

  if (DumpKernelASM)
    outs() << Assembly << "\n";

  GPUModule.release();
  KernelIDs.clear();

  return Assembly;
}

namespace {
class PPCGCodeGeneration : public ScopPass {
public:
  static char ID;

  /// The scop that is currently processed.
  Scop *S;

  LoopInfo *LI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  const DataLayout *DL;
  RegionInfo *RI;

  PPCGCodeGeneration() : ScopPass(ID) {}

  /// Construct compilation options for PPCG.
  ///
  /// @returns The compilation options.
  ppcg_options *createPPCGOptions() {
    auto DebugOptions =
        (ppcg_debug_options *)malloc(sizeof(ppcg_debug_options));
    auto Options = (ppcg_options *)malloc(sizeof(ppcg_options));

    DebugOptions->dump_schedule_constraints = false;
    DebugOptions->dump_schedule = false;
    DebugOptions->dump_final_schedule = false;
    DebugOptions->dump_sizes = false;
    DebugOptions->verbose = false;

    Options->debug = DebugOptions;

    Options->reschedule = true;
    Options->scale_tile_loops = false;
    Options->wrap = false;

    Options->non_negative_parameters = false;
    Options->ctx = nullptr;
    Options->sizes = nullptr;

    Options->tile_size = 32;

    Options->use_private_memory = PrivateMemory;
    Options->use_shared_memory = SharedMemory;
    Options->max_shared_memory = 48 * 1024;

    Options->target = PPCG_TARGET_CUDA;
    Options->openmp = false;
    Options->linearize_device_arrays = true;
    Options->live_range_reordering = false;

    Options->opencl_compiler_options = nullptr;
    Options->opencl_use_gpu = false;
    Options->opencl_n_include_file = 0;
    Options->opencl_include_files = nullptr;
    Options->opencl_print_kernel_types = false;
    Options->opencl_embed_kernel_code = false;

    Options->save_schedule_file = nullptr;
    Options->load_schedule_file = nullptr;

    return Options;
  }

  /// Get a tagged access relation containing all accesses of type @p AccessTy.
  ///
  /// Instead of a normal access of the form:
  ///
  ///   Stmt[i,j,k] -> Array[f_0(i,j,k), f_1(i,j,k)]
  ///
  /// a tagged access has the form
  ///
  ///   [Stmt[i,j,k] -> id[]] -> Array[f_0(i,j,k), f_1(i,j,k)]
  ///
  /// where 'id' is an additional space that references the memory access that
  /// triggered the access.
  ///
  /// @param AccessTy The type of the memory accesses to collect.
  ///
  /// @return The relation describing all tagged memory accesses.
  isl_union_map *getTaggedAccesses(enum MemoryAccess::AccessType AccessTy) {
    isl_union_map *Accesses = isl_union_map_empty(S->getParamSpace());

    for (auto &Stmt : *S)
      for (auto &Acc : Stmt)
        if (Acc->getType() == AccessTy) {
          isl_map *Relation = Acc->getAccessRelation();
          Relation = isl_map_intersect_domain(Relation, Stmt.getDomain());

          isl_space *Space = isl_map_get_space(Relation);
          Space = isl_space_range(Space);
          Space = isl_space_from_range(Space);
          Space = isl_space_set_tuple_id(Space, isl_dim_in, Acc->getId());
          isl_map *Universe = isl_map_universe(Space);
          Relation = isl_map_domain_product(Relation, Universe);
          Accesses = isl_union_map_add_map(Accesses, Relation);
        }

    return Accesses;
  }

  /// Get the set of all read accesses, tagged with the access id.
  ///
  /// @see getTaggedAccesses
  isl_union_map *getTaggedReads() {
    return getTaggedAccesses(MemoryAccess::READ);
  }

  /// Get the set of all may (and must) accesses, tagged with the access id.
  ///
  /// @see getTaggedAccesses
  isl_union_map *getTaggedMayWrites() {
    return isl_union_map_union(getTaggedAccesses(MemoryAccess::MAY_WRITE),
                               getTaggedAccesses(MemoryAccess::MUST_WRITE));
  }

  /// Get the set of all must accesses, tagged with the access id.
  ///
  /// @see getTaggedAccesses
  isl_union_map *getTaggedMustWrites() {
    return getTaggedAccesses(MemoryAccess::MUST_WRITE);
  }

  /// Collect parameter and array names as isl_ids.
  ///
  /// To reason about the different parameters and arrays used, ppcg requires
  /// a list of all isl_ids in use. As PPCG traditionally performs
  /// source-to-source compilation each of these isl_ids is mapped to the
  /// expression that represents it. As we do not have a corresponding
  /// expression in Polly, we just map each id to a 'zero' expression to match
  /// the data format that ppcg expects.
  ///
  /// @returns Retun a map from collected ids to 'zero' ast expressions.
  __isl_give isl_id_to_ast_expr *getNames() {
    auto *Names = isl_id_to_ast_expr_alloc(
        S->getIslCtx(),
        S->getNumParams() + std::distance(S->array_begin(), S->array_end()));
    auto *Zero = isl_ast_expr_from_val(isl_val_zero(S->getIslCtx()));
    auto *Space = S->getParamSpace();

    for (int I = 0, E = S->getNumParams(); I < E; ++I) {
      isl_id *Id = isl_space_get_dim_id(Space, isl_dim_param, I);
      Names = isl_id_to_ast_expr_set(Names, Id, isl_ast_expr_copy(Zero));
    }

    for (auto &Array : S->arrays()) {
      auto Id = Array->getBasePtrId();
      Names = isl_id_to_ast_expr_set(Names, Id, isl_ast_expr_copy(Zero));
    }

    isl_space_free(Space);
    isl_ast_expr_free(Zero);

    return Names;
  }

  /// Create a new PPCG scop from the current scop.
  ///
  /// The PPCG scop is initialized with data from the current polly::Scop. From
  /// this initial data, the data-dependences in the PPCG scop are initialized.
  /// We do not use Polly's dependence analysis for now, to ensure we match
  /// the PPCG default behaviour more closely.
  ///
  /// @returns A new ppcg scop.
  ppcg_scop *createPPCGScop() {
    auto PPCGScop = (ppcg_scop *)malloc(sizeof(ppcg_scop));

    PPCGScop->options = createPPCGOptions();

    PPCGScop->start = 0;
    PPCGScop->end = 0;

    PPCGScop->context = S->getContext();
    PPCGScop->domain = S->getDomains();
    PPCGScop->call = nullptr;
    PPCGScop->tagged_reads = getTaggedReads();
    PPCGScop->reads = S->getReads();
    PPCGScop->live_in = nullptr;
    PPCGScop->tagged_may_writes = getTaggedMayWrites();
    PPCGScop->may_writes = S->getWrites();
    PPCGScop->tagged_must_writes = getTaggedMustWrites();
    PPCGScop->must_writes = S->getMustWrites();
    PPCGScop->live_out = nullptr;
    PPCGScop->tagged_must_kills = isl_union_map_empty(S->getParamSpace());
    PPCGScop->tagger = nullptr;

    PPCGScop->independence = nullptr;
    PPCGScop->dep_flow = nullptr;
    PPCGScop->tagged_dep_flow = nullptr;
    PPCGScop->dep_false = nullptr;
    PPCGScop->dep_forced = nullptr;
    PPCGScop->dep_order = nullptr;
    PPCGScop->tagged_dep_order = nullptr;

    PPCGScop->schedule = S->getScheduleTree();
    PPCGScop->names = getNames();

    PPCGScop->pet = nullptr;

    compute_tagger(PPCGScop);
    compute_dependences(PPCGScop);

    return PPCGScop;
  }

  /// Collect the array acesses in a statement.
  ///
  /// @param Stmt The statement for which to collect the accesses.
  ///
  /// @returns A list of array accesses.
  gpu_stmt_access *getStmtAccesses(ScopStmt &Stmt) {
    gpu_stmt_access *Accesses = nullptr;

    for (MemoryAccess *Acc : Stmt) {
      auto Access = isl_alloc_type(S->getIslCtx(), struct gpu_stmt_access);
      Access->read = Acc->isRead();
      Access->write = Acc->isWrite();
      Access->access = Acc->getAccessRelation();
      isl_space *Space = isl_map_get_space(Access->access);
      Space = isl_space_range(Space);
      Space = isl_space_from_range(Space);
      Space = isl_space_set_tuple_id(Space, isl_dim_in, Acc->getId());
      isl_map *Universe = isl_map_universe(Space);
      Access->tagged_access =
          isl_map_domain_product(Acc->getAccessRelation(), Universe);
      Access->exact_write = !Acc->isMayWrite();
      Access->ref_id = Acc->getId();
      Access->next = Accesses;
      Access->n_index = Acc->getScopArrayInfo()->getNumberOfDimensions();
      Accesses = Access;
    }

    return Accesses;
  }

  /// Collect the list of GPU statements.
  ///
  /// Each statement has an id, a pointer to the underlying data structure,
  /// as well as a list with all memory accesses.
  ///
  /// TODO: Initialize the list of memory accesses.
  ///
  /// @returns A linked-list of statements.
  gpu_stmt *getStatements() {
    gpu_stmt *Stmts = isl_calloc_array(S->getIslCtx(), struct gpu_stmt,
                                       std::distance(S->begin(), S->end()));

    int i = 0;
    for (auto &Stmt : *S) {
      gpu_stmt *GPUStmt = &Stmts[i];

      GPUStmt->id = Stmt.getDomainId();

      // We use the pet stmt pointer to keep track of the Polly statements.
      GPUStmt->stmt = (pet_stmt *)&Stmt;
      GPUStmt->accesses = getStmtAccesses(Stmt);
      i++;
    }

    return Stmts;
  }

  /// Derive the extent of an array.
  ///
  /// The extent of an array is the set of elements that are within the
  /// accessed array. For the inner dimensions, the extent constraints are
  /// 0 and the size of the corresponding array dimension. For the first
  /// (outermost) dimension, the extent constraints are the minimal and maximal
  /// subscript value for the first dimension.
  ///
  /// @param Array The array to derive the extent for.
  ///
  /// @returns An isl_set describing the extent of the array.
  __isl_give isl_set *getExtent(ScopArrayInfo *Array) {
    unsigned NumDims = Array->getNumberOfDimensions();
    isl_union_map *Accesses = S->getAccesses();
    Accesses = isl_union_map_intersect_domain(Accesses, S->getDomains());
    Accesses = isl_union_map_detect_equalities(Accesses);
    isl_union_set *AccessUSet = isl_union_map_range(Accesses);
    AccessUSet = isl_union_set_coalesce(AccessUSet);
    AccessUSet = isl_union_set_detect_equalities(AccessUSet);
    AccessUSet = isl_union_set_coalesce(AccessUSet);

    if (isl_union_set_is_empty(AccessUSet)) {
      isl_union_set_free(AccessUSet);
      return isl_set_empty(Array->getSpace());
    }

    if (Array->getNumberOfDimensions() == 0) {
      isl_union_set_free(AccessUSet);
      return isl_set_universe(Array->getSpace());
    }

    isl_set *AccessSet =
        isl_union_set_extract_set(AccessUSet, Array->getSpace());

    isl_union_set_free(AccessUSet);
    isl_local_space *LS = isl_local_space_from_space(Array->getSpace());

    isl_pw_aff *Val =
        isl_pw_aff_from_aff(isl_aff_var_on_domain(LS, isl_dim_set, 0));

    isl_pw_aff *OuterMin = isl_set_dim_min(isl_set_copy(AccessSet), 0);
    isl_pw_aff *OuterMax = isl_set_dim_max(AccessSet, 0);
    OuterMin = isl_pw_aff_add_dims(OuterMin, isl_dim_in,
                                   isl_pw_aff_dim(Val, isl_dim_in));
    OuterMax = isl_pw_aff_add_dims(OuterMax, isl_dim_in,
                                   isl_pw_aff_dim(Val, isl_dim_in));
    OuterMin =
        isl_pw_aff_set_tuple_id(OuterMin, isl_dim_in, Array->getBasePtrId());
    OuterMax =
        isl_pw_aff_set_tuple_id(OuterMax, isl_dim_in, Array->getBasePtrId());

    isl_set *Extent = isl_set_universe(Array->getSpace());

    Extent = isl_set_intersect(
        Extent, isl_pw_aff_le_set(OuterMin, isl_pw_aff_copy(Val)));
    Extent = isl_set_intersect(Extent, isl_pw_aff_ge_set(OuterMax, Val));

    for (unsigned i = 1; i < NumDims; ++i)
      Extent = isl_set_lower_bound_si(Extent, isl_dim_set, i, 0);

    for (unsigned i = 1; i < NumDims; ++i) {
      isl_pw_aff *PwAff =
          const_cast<isl_pw_aff *>(Array->getDimensionSizePw(i));
      isl_pw_aff *Val = isl_pw_aff_from_aff(isl_aff_var_on_domain(
          isl_local_space_from_space(Array->getSpace()), isl_dim_set, i));
      PwAff = isl_pw_aff_add_dims(PwAff, isl_dim_in,
                                  isl_pw_aff_dim(Val, isl_dim_in));
      PwAff = isl_pw_aff_set_tuple_id(PwAff, isl_dim_in,
                                      isl_pw_aff_get_tuple_id(Val, isl_dim_in));
      auto *Set = isl_pw_aff_gt_set(PwAff, Val);
      Extent = isl_set_intersect(Set, Extent);
    }

    return Extent;
  }

  /// Derive the bounds of an array.
  ///
  /// For the first dimension we derive the bound of the array from the extent
  /// of this dimension. For inner dimensions we obtain their size directly from
  /// ScopArrayInfo.
  ///
  /// @param PPCGArray The array to compute bounds for.
  /// @param Array The polly array from which to take the information.
  void setArrayBounds(gpu_array_info &PPCGArray, ScopArrayInfo *Array) {
    if (PPCGArray.n_index > 0) {
      if (isl_set_is_empty(PPCGArray.extent)) {
        isl_set *Dom = isl_set_copy(PPCGArray.extent);
        isl_local_space *LS = isl_local_space_from_space(
            isl_space_params(isl_set_get_space(Dom)));
        isl_set_free(Dom);
        isl_aff *Zero = isl_aff_zero_on_domain(LS);
        PPCGArray.bound[0] = isl_pw_aff_from_aff(Zero);
      } else {
        isl_set *Dom = isl_set_copy(PPCGArray.extent);
        Dom = isl_set_project_out(Dom, isl_dim_set, 1, PPCGArray.n_index - 1);
        isl_pw_aff *Bound = isl_set_dim_max(isl_set_copy(Dom), 0);
        isl_set_free(Dom);
        Dom = isl_pw_aff_domain(isl_pw_aff_copy(Bound));
        isl_local_space *LS =
            isl_local_space_from_space(isl_set_get_space(Dom));
        isl_aff *One = isl_aff_zero_on_domain(LS);
        One = isl_aff_add_constant_si(One, 1);
        Bound = isl_pw_aff_add(Bound, isl_pw_aff_alloc(Dom, One));
        Bound = isl_pw_aff_gist(Bound, S->getContext());
        PPCGArray.bound[0] = Bound;
      }
    }

    for (unsigned i = 1; i < PPCGArray.n_index; ++i) {
      isl_pw_aff *Bound = Array->getDimensionSizePw(i);
      auto LS = isl_pw_aff_get_domain_space(Bound);
      auto Aff = isl_multi_aff_zero(LS);
      Bound = isl_pw_aff_pullback_multi_aff(Bound, Aff);
      PPCGArray.bound[i] = Bound;
    }
  }

  /// Create the arrays for @p PPCGProg.
  ///
  /// @param PPCGProg The program to compute the arrays for.
  void createArrays(gpu_prog *PPCGProg) {
    int i = 0;
    for (auto &Array : S->arrays()) {
      std::string TypeName;
      raw_string_ostream OS(TypeName);

      OS << *Array->getElementType();
      TypeName = OS.str();

      gpu_array_info &PPCGArray = PPCGProg->array[i];

      PPCGArray.space = Array->getSpace();
      PPCGArray.type = strdup(TypeName.c_str());
      PPCGArray.size = Array->getElementType()->getPrimitiveSizeInBits() / 8;
      PPCGArray.name = strdup(Array->getName().c_str());
      PPCGArray.extent = nullptr;
      PPCGArray.n_index = Array->getNumberOfDimensions();
      PPCGArray.bound =
          isl_alloc_array(S->getIslCtx(), isl_pw_aff *, PPCGArray.n_index);
      PPCGArray.extent = getExtent(Array);
      PPCGArray.n_ref = 0;
      PPCGArray.refs = nullptr;
      PPCGArray.accessed = true;
      PPCGArray.read_only_scalar = false;
      PPCGArray.has_compound_element = false;
      PPCGArray.local = false;
      PPCGArray.declare_local = false;
      PPCGArray.global = false;
      PPCGArray.linearize = false;
      PPCGArray.dep_order = nullptr;
      PPCGArray.user = Array;

      setArrayBounds(PPCGArray, Array);
      i++;

      collect_references(PPCGProg, &PPCGArray);
    }
  }

  /// Create an identity map between the arrays in the scop.
  ///
  /// @returns An identity map between the arrays in the scop.
  isl_union_map *getArrayIdentity() {
    isl_union_map *Maps = isl_union_map_empty(S->getParamSpace());

    for (auto &Array : S->arrays()) {
      isl_space *Space = Array->getSpace();
      Space = isl_space_map_from_set(Space);
      isl_map *Identity = isl_map_identity(Space);
      Maps = isl_union_map_add_map(Maps, Identity);
    }

    return Maps;
  }

  /// Create a default-initialized PPCG GPU program.
  ///
  /// @returns A new gpu grogram description.
  gpu_prog *createPPCGProg(ppcg_scop *PPCGScop) {

    if (!PPCGScop)
      return nullptr;

    auto PPCGProg = isl_calloc_type(S->getIslCtx(), struct gpu_prog);

    PPCGProg->ctx = S->getIslCtx();
    PPCGProg->scop = PPCGScop;
    PPCGProg->context = isl_set_copy(PPCGScop->context);
    PPCGProg->read = isl_union_map_copy(PPCGScop->reads);
    PPCGProg->may_write = isl_union_map_copy(PPCGScop->may_writes);
    PPCGProg->must_write = isl_union_map_copy(PPCGScop->must_writes);
    PPCGProg->tagged_must_kill =
        isl_union_map_copy(PPCGScop->tagged_must_kills);
    PPCGProg->to_inner = getArrayIdentity();
    PPCGProg->to_outer = getArrayIdentity();
    PPCGProg->any_to_outer = nullptr;
    PPCGProg->array_order = nullptr;
    PPCGProg->n_stmts = std::distance(S->begin(), S->end());
    PPCGProg->stmts = getStatements();
    PPCGProg->n_array = std::distance(S->array_begin(), S->array_end());
    PPCGProg->array = isl_calloc_array(S->getIslCtx(), struct gpu_array_info,
                                       PPCGProg->n_array);

    createArrays(PPCGProg);

    PPCGProg->may_persist = compute_may_persist(PPCGProg);

    return PPCGProg;
  }

  struct PrintGPUUserData {
    struct cuda_info *CudaInfo;
    struct gpu_prog *PPCGProg;
    std::vector<ppcg_kernel *> Kernels;
  };

  /// Print a user statement node in the host code.
  ///
  /// We use ppcg's printing facilities to print the actual statement and
  /// additionally build up a list of all kernels that are encountered in the
  /// host ast.
  ///
  /// @param P The printer to print to
  /// @param Options The printing options to use
  /// @param Node The node to print
  /// @param User A user pointer to carry additional data. This pointer is
  ///             expected to be of type PrintGPUUserData.
  ///
  /// @returns A printer to which the output has been printed.
  static __isl_give isl_printer *
  printHostUser(__isl_take isl_printer *P,
                __isl_take isl_ast_print_options *Options,
                __isl_take isl_ast_node *Node, void *User) {
    auto Data = (struct PrintGPUUserData *)User;
    auto Id = isl_ast_node_get_annotation(Node);

    if (Id) {
      bool IsUser = !strcmp(isl_id_get_name(Id), "user");

      // If this is a user statement, format it ourselves as ppcg would
      // otherwise try to call pet functionality that is not available in
      // Polly.
      if (IsUser) {
        P = isl_printer_start_line(P);
        P = isl_printer_print_ast_node(P, Node);
        P = isl_printer_end_line(P);
        isl_id_free(Id);
        isl_ast_print_options_free(Options);
        return P;
      }

      auto Kernel = (struct ppcg_kernel *)isl_id_get_user(Id);
      isl_id_free(Id);
      Data->Kernels.push_back(Kernel);
    }

    return print_host_user(P, Options, Node, User);
  }

  /// Print C code corresponding to the control flow in @p Kernel.
  ///
  /// @param Kernel The kernel to print
  void printKernel(ppcg_kernel *Kernel) {
    auto *P = isl_printer_to_str(S->getIslCtx());
    P = isl_printer_set_output_format(P, ISL_FORMAT_C);
    auto *Options = isl_ast_print_options_alloc(S->getIslCtx());
    P = isl_ast_node_print(Kernel->tree, P, Options);
    char *String = isl_printer_get_str(P);
    printf("%s\n", String);
    free(String);
    isl_printer_free(P);
  }

  /// Print C code corresponding to the GPU code described by @p Tree.
  ///
  /// @param Tree An AST describing GPU code
  /// @param PPCGProg The PPCG program from which @Tree has been constructed.
  void printGPUTree(isl_ast_node *Tree, gpu_prog *PPCGProg) {
    auto *P = isl_printer_to_str(S->getIslCtx());
    P = isl_printer_set_output_format(P, ISL_FORMAT_C);

    PrintGPUUserData Data;
    Data.PPCGProg = PPCGProg;

    auto *Options = isl_ast_print_options_alloc(S->getIslCtx());
    Options =
        isl_ast_print_options_set_print_user(Options, printHostUser, &Data);
    P = isl_ast_node_print(Tree, P, Options);
    char *String = isl_printer_get_str(P);
    printf("# host\n");
    printf("%s\n", String);
    free(String);
    isl_printer_free(P);

    for (auto Kernel : Data.Kernels) {
      printf("# kernel%d\n", Kernel->id);
      printKernel(Kernel);
    }
  }

  // Generate a GPU program using PPCG.
  //
  // GPU mapping consists of multiple steps:
  //
  //  1) Compute new schedule for the program.
  //  2) Map schedule to GPU (TODO)
  //  3) Generate code for new schedule (TODO)
  //
  // We do not use here the Polly ScheduleOptimizer, as the schedule optimizer
  // is mostly CPU specific. Instead, we use PPCG's GPU code generation
  // strategy directly from this pass.
  gpu_gen *generateGPU(ppcg_scop *PPCGScop, gpu_prog *PPCGProg) {

    auto PPCGGen = isl_calloc_type(S->getIslCtx(), struct gpu_gen);

    PPCGGen->ctx = S->getIslCtx();
    PPCGGen->options = PPCGScop->options;
    PPCGGen->print = nullptr;
    PPCGGen->print_user = nullptr;
    PPCGGen->build_ast_expr = &pollyBuildAstExprForStmt;
    PPCGGen->prog = PPCGProg;
    PPCGGen->tree = nullptr;
    PPCGGen->types.n = 0;
    PPCGGen->types.name = nullptr;
    PPCGGen->sizes = nullptr;
    PPCGGen->used_sizes = nullptr;
    PPCGGen->kernel_id = 0;

    // Set scheduling strategy to same strategy PPCG is using.
    isl_options_set_schedule_outer_coincidence(PPCGGen->ctx, true);
    isl_options_set_schedule_maximize_band_depth(PPCGGen->ctx, true);
    isl_options_set_schedule_whole_component(PPCGGen->ctx, false);

    isl_schedule *Schedule = get_schedule(PPCGGen);

    int has_permutable = has_any_permutable_node(Schedule);

    if (!has_permutable || has_permutable < 0) {
      Schedule = isl_schedule_free(Schedule);
    } else {
      Schedule = map_to_device(PPCGGen, Schedule);
      PPCGGen->tree = generate_code(PPCGGen, isl_schedule_copy(Schedule));
    }

    if (DumpSchedule) {
      isl_printer *P = isl_printer_to_str(S->getIslCtx());
      P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
      P = isl_printer_print_str(P, "Schedule\n");
      P = isl_printer_print_str(P, "========\n");
      if (Schedule)
        P = isl_printer_print_schedule(P, Schedule);
      else
        P = isl_printer_print_str(P, "No schedule found\n");

      printf("%s\n", isl_printer_get_str(P));
      isl_printer_free(P);
    }

    if (DumpCode) {
      printf("Code\n");
      printf("====\n");
      if (PPCGGen->tree)
        printGPUTree(PPCGGen->tree, PPCGProg);
      else
        printf("No code generated\n");
    }

    isl_schedule_free(Schedule);

    return PPCGGen;
  }

  /// Free gpu_gen structure.
  ///
  /// @param PPCGGen The ppcg_gen object to free.
  void freePPCGGen(gpu_gen *PPCGGen) {
    isl_ast_node_free(PPCGGen->tree);
    isl_union_map_free(PPCGGen->sizes);
    isl_union_map_free(PPCGGen->used_sizes);
    free(PPCGGen);
  }

  /// Free the options in the ppcg scop structure.
  ///
  /// ppcg is not freeing these options for us. To avoid leaks we do this
  /// ourselves.
  ///
  /// @param PPCGScop The scop referencing the options to free.
  void freeOptions(ppcg_scop *PPCGScop) {
    free(PPCGScop->options->debug);
    PPCGScop->options->debug = nullptr;
    free(PPCGScop->options);
    PPCGScop->options = nullptr;
  }

  /// Generate code for a given GPU AST described by @p Root.
  ///
  /// @param Root An isl_ast_node pointing to the root of the GPU AST.
  /// @param Prog The GPU Program to generate code for.
  void generateCode(__isl_take isl_ast_node *Root, gpu_prog *Prog) {
    ScopAnnotator Annotator;
    Annotator.buildAliasScopes(*S);

    Region *R = &S->getRegion();

    simplifyRegion(R, DT, LI, RI);

    BasicBlock *EnteringBB = R->getEnteringBlock();

    PollyIRBuilder Builder = createPollyIRBuilder(EnteringBB, Annotator);

    GPUNodeBuilder NodeBuilder(Builder, Annotator, this, *DL, *LI, *SE, *DT, *S,
                               Prog);

    // Only build the run-time condition and parameters _after_ having
    // introduced the conditional branch. This is important as the conditional
    // branch will guard the original scop from new induction variables that
    // the SCEVExpander may introduce while code generating the parameters and
    // which may introduce scalar dependences that prevent us from correctly
    // code generating this scop.
    BasicBlock *StartBlock =
        executeScopConditionally(*S, this, Builder.getTrue());

    // TODO: Handle LICM
    auto SplitBlock = StartBlock->getSinglePredecessor();
    Builder.SetInsertPoint(SplitBlock->getTerminator());
    NodeBuilder.addParameters(S->getContext());

    isl_ast_build *Build = isl_ast_build_alloc(S->getIslCtx());
    isl_ast_expr *Condition = IslAst::buildRunCondition(S, Build);
    isl_ast_build_free(Build);

    Value *RTC = NodeBuilder.createRTC(Condition);
    Builder.GetInsertBlock()->getTerminator()->setOperand(0, RTC);

    Builder.SetInsertPoint(&*StartBlock->begin());

    NodeBuilder.initializeAfterRTH();
    NodeBuilder.create(Root);
    NodeBuilder.finalize();

    if (!NodeBuilder.BuildSuccessful)
      SplitBlock->getTerminator()->setOperand(0, Builder.getFalse());
  }

  bool runOnScop(Scop &CurrentScop) override {
    S = &CurrentScop;
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    DL = &S->getRegion().getEntry()->getParent()->getParent()->getDataLayout();
    RI = &getAnalysis<RegionInfoPass>().getRegionInfo();

    // We currently do not support scops with invariant loads.
    if (S->hasInvariantAccesses())
      return false;

    auto PPCGScop = createPPCGScop();
    auto PPCGProg = createPPCGProg(PPCGScop);
    auto PPCGGen = generateGPU(PPCGScop, PPCGProg);

    if (PPCGGen->tree)
      generateCode(isl_ast_node_copy(PPCGGen->tree), PPCGProg);

    freeOptions(PPCGScop);
    freePPCGGen(PPCGGen);
    gpu_prog_free(PPCGProg);
    ppcg_scop_free(PPCGScop);

    return true;
  }

  void printScop(raw_ostream &, Scop &) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<PostDominatorTreeWrapperPass>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addPreserved<SCEVAAWrapperPass>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
    AU.addPreserved<RegionInfoPass>();
    AU.addPreserved<ScopInfoRegionPass>();
  }
};
}

char PPCGCodeGeneration::ID = 1;

Pass *polly::createPPCGCodeGenerationPass() { return new PPCGCodeGeneration(); }

INITIALIZE_PASS_BEGIN(PPCGCodeGeneration, "polly-codegen-ppcg",
                      "Polly - Apply PPCG translation to SCOP", false, false)
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(PPCGCodeGeneration, "polly-codegen-ppcg",
                    "Polly - Apply PPCG translation to SCOP", false, false)
