//===-- LanguageAliases.inc - Language runtime symbol aliases -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LLVM offload kernel runtime implements the shared __llvm* entry points
// once. These wrappers expose the CUDA/HIP ABI names, such as
// __cudaRegisterFunction and __hipPopCallConfiguration, as forwarding entry
// points to the shared implementations.
//
//===----------------------------------------------------------------------===//

// This file intentionally has no include guard. It is included once for each
// LANGUAGE value to emit each language's alias definitions.

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

#define LA_IMPL2(PREFIX, L, NAME) PREFIX##L##NAME
#define LA_IMPL1(PREFIX, L, NAME) LA_IMPL2(PREFIX, L, NAME)
#define LANGUAGE_NAME(PREFIX, NAME) LA_IMPL1(PREFIX, LANGUAGE, NAME)

extern "C" void
LANGUAGE_NAME(__, RegisterFunction)(const char *Binary, const char *KernelID,
                                    char *KernelName, const char *KernelName1,
                                    int ThreadLimit, uint3 *Tid, uint3 *Bid,
                                    dim3 *BlockDim, dim3 *GridDim, int *WSize) {
  __llvmRegisterFunction(Binary, KernelID, KernelName, KernelName1, ThreadLimit,
                         Tid, Bid, BlockDim, GridDim, WSize);
}

extern "C" void LANGUAGE_NAME(__, RegisterVar)(void **Data, char *HostVar,
                                               char *DeviceAddress,
                                               const char *DeviceName, int Ext,
                                               int Size, int Constant,
                                               int Global) {
  __llvmRegisterVar(Data, HostVar, DeviceAddress, DeviceName, Ext, Size,
                    Constant, Global);
}

extern "C" void LANGUAGE_NAME(__,
                              RegisterManagedVar)(void **Data, char *HostVar,
                                                  char *DeviceAddress,
                                                  const char *DeviceName,
                                                  size_t Size, unsigned Align) {
  __llvmRegisterManagedVar(Data, HostVar, DeviceAddress, DeviceName, Size,
                           Align);
}

extern "C" void LANGUAGE_NAME(__, RegisterSurface)(
    void **Data, const struct surfaceReference *SurfRef, const void **DevPtr,
    const char *Name, int Dim, int Ext) {
  __llvmRegisterSurface(Data, SurfRef, DevPtr, Name, Dim, Ext);
}

extern "C" void LANGUAGE_NAME(__, RegisterTexture)(
    void **Data, const struct textureReference *TexRef, const void **DevPtr,
    const char *Name, int Dim, int Norm, int Ext) {
  __llvmRegisterTexture(Data, TexRef, DevPtr, Name, Dim, Norm, Ext);
}

#undef LANGUAGE_NAME
#undef LA_IMPL1
#undef LA_IMPL2
