//===- PTXGenerator.h - IR helper to create GPGPU LLVM-IR -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create GPGPU parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_CODEGEN_PTXGENERATOR_H
#define POLLY_CODEGEN_PTXGENERATOR_H

#include "polly/Config/config.h"

#ifdef GPU_CODEGEN
#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/SetVector.h"

#include <map>

namespace llvm {
class Value;
class Pass;
class BasicBlock;
}

namespace polly {
using namespace llvm;

class PTXGenerator {
public:
  typedef std::map<Value *, Value *> ValueToValueMapTy;

  PTXGenerator(IRBuilder<> &Builder, Pass *P, const std::string &Triple);

  /// @brief Create a GPGPU parallel loop.
  ///
  /// @param UsedValues   A set of LLVM-IR Values that should be available to
  ///                     the new loop body.
  /// @param OriginalIVS  The new values of the original induction variables.
  /// @param VMap         This map is filled by createParallelLoop(). It
  ///                     maps the values in UsedValues to Values through which
  ///                     their content is available within the loop body.
  /// @param LoopBody     A pointer to an iterator that is set to point to the
  ///                     body of the created loop. It should be used to insert
  ///                     instructions that form the actual loop body.
  void startGeneration(SetVector<Value *> &UsedValues,
                       SetVector<Value *> &OriginalIVS, ValueToValueMapTy &VMap,
                       BasicBlock::iterator *LoopBody);

  /// @brief Execute the post-operations to build a GPGPU parallel loop.
  ///
  void finishGeneration(Function *SubFunction);

  /// @brief Set the parameters for launching PTX kernel.
  ///
  /// @param GridW    A value of the width of a GPU grid.
  /// @param GridH    A value of the height of a GPU grid.
  /// @param BlockW   A value of the width of a GPU block.
  /// @param BlockH   A value of the height of a GPU block.
  void setLaunchingParameters(int GridW, int GridH, int BlockW, int BlockH) {
    GridWidth = GridW;
    GridHeight = GridH;
    BlockWidth = BlockW;
    BlockHeight = BlockH;
  }

  /// @brief Set the size of the output array.
  ///
  /// This size is used to allocate memory on the device and the host.
  ///
  /// @param Bytes        Output array size in bytes.
  void setOutputBytes(unsigned Bytes) { OutputBytes = Bytes; }

private:
  IRBuilder<> &Builder;
  Pass *P;

  /// @brief The target triple of the device.
  const std::string &GPUTriple;

  ///@brief Parameters used for launching PTX kernel.
  int GridWidth, GridHeight, BlockWidth, BlockHeight;

  /// @brief Size of the output array in bytes.
  unsigned OutputBytes;

  /// @brief Polly's GPU data types.
  StructType *ContextTy, *ModuleTy, *KernelTy, *DeviceTy, *DevDataTy, *EventTy;

  void InitializeGPUDataTypes();
  IntegerType *getInt64Type();           // i64
  PointerType *getI8PtrType();           // char *
  PointerType *getPtrI8PtrType();        // char **
  PointerType *getFloatPtrType();        // float *
  PointerType *getGPUContextPtrType();   // %struct.PollyGPUContextT *
  PointerType *getGPUModulePtrType();    // %struct.PollyGPUModuleT *
  PointerType *getGPUDevicePtrType();    // %struct.PollyGPUDeviceT *
  PointerType *getPtrGPUDevicePtrType(); // %struct.PollyGPUDevicePtrT *
  PointerType *getGPUFunctionPtrType();  // %struct.PollyGPUFunctionT *
  PointerType *getGPUEventPtrType();     // %struct.PollyGPUEventT *

  Module *getModule();

  /// @brief Create the kernel string containing LLVM IR.
  ///
  /// @param SubFunction  A pointer to the device code function.
  /// @return             A global string variable containing the LLVM IR codes
  //                      of the SubFunction.
  Value *createPTXKernelFunction(Function *SubFunction);

  /// @brief Get the entry name of the device kernel function.
  ///
  /// @param SubFunction  A pointer to the device code function.
  /// @return             A global string variable containing the entry name of
  ///                     the SubFunction.
  Value *getPTXKernelEntryName(Function *SubFunction);

  void createCallInitDevice(Value *Context, Value *Device);
  void createCallGetPTXModule(Value *Buffer, Value *Module);
  void createCallGetPTXKernelEntry(Value *Entry, Value *Module, Value *Kernel);
  void createCallAllocateMemoryForHostAndDevice(Value *HostData,
                                                Value *DeviceData, Value *Size);
  void createCallCopyFromHostToDevice(Value *DeviceData, Value *HostData,
                                      Value *Size);
  void createCallCopyFromDeviceToHost(Value *HostData, Value *DeviceData,
                                      Value *Size);
  void createCallSetKernelParameters(Value *Kernel, Value *BlockWidth,
                                     Value *BlockHeight, Value *DeviceData);
  void createCallLaunchKernel(Value *Kernel, Value *GridWidth,
                              Value *GridHeight);
  void createCallStartTimerByCudaEvent(Value *StartEvent, Value *StopEvent);
  void createCallStopTimerByCudaEvent(Value *StartEvent, Value *StopEvent,
                                      Value *Timer);
  void createCallCleanupGPGPUResources(Value *HostData, Value *DeviceData,
                                       Value *Module, Value *Context,
                                       Value *Kernel);

  /// @brief Create the CUDA subfunction.
  ///
  /// @param UsedValues   A set of LLVM-IR Values that should be available to
  ///                     the new loop body.
  /// @param VMap         This map that is filled by createSubfunction(). It
  ///                     maps the values in UsedValues to Values through which
  ///                     their content is available within the loop body.
  /// @param OriginalIVS  The new values of the original induction variables.
  /// @param SubFunction  The newly created SubFunction is returned here.
  void createSubfunction(SetVector<Value *> &UsedValues,
                         SetVector<Value *> &OriginalIVS,
                         ValueToValueMapTy &VMap, Function **SubFunction);

  /// @brief Create the definition of the CUDA subfunction.
  ///
  /// @param NumArgs      The number of parameters of this subfunction. This is
  ///                     usually set to the number of memory accesses which
  ///                     will be copied from host to device.
  Function *createSubfunctionDefinition(int NumArgs);

  /// @brief Extract all the ptx related subfunctions into a new module.
  ///
  /// @param M            Current module.
  /// @return             The generated module containing only gpu related
  ///                     subfunctions.
  Module *extractPTXFunctionsFromModule(const Module *M);

  /// @brief Get the Value of CUDA block width.
  Value *getCUDABlockWidth();

  /// @brief Get the Value of CUDA block height.
  Value *getCUDABlockHeight();

  /// @brief Get the Value of CUDA Gird width.
  Value *getCUDAGridWidth();

  /// @brief Get the Value of CUDA grid height.
  Value *getCUDAGridHeight();

  /// @brief Get the Value of the bytes of the output array.
  Value *getOutputArraySizeInBytes();

  /// @brief Erase the ptx-related subfunctions and declarations.
  ///
  /// @param SubFunction  A pointer to the device code function.
  void eraseUnusedFunctions(Function *SubFunction);
};
}      // end namespace polly
#endif /* GPU_CODEGEN */
#endif /* POLLY_CODEGEN_PTXGENERATOR_H */
