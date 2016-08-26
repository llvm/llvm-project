/******************************************************************************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is dual licensed under the MIT and the University of Illinois    */
/* Open Source License. See LICENSE.TXT for details.                          */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file defines GPUJIT.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef GPUJIT_H_
#define GPUJIT_H_

/*
 * The following demostrates how we can use the GPURuntime library to
 * execute a GPU kernel.
 *
 * char KernelString[] = "\n\
 *   .version 1.4\n\
 *   .target sm_10, map_f64_to_f32\n\
 *   .entry _Z8myKernelPi (\n\
 *   .param .u64 __cudaparm__Z8myKernelPi_data)\n\
 *   {\n\
 *     .reg .u16 %rh<4>;\n\
 *     .reg .u32 %r<5>;\n\
 *     .reg .u64 %rd<6>;\n\
 *     cvt.u32.u16     %r1, %tid.x;\n\
 *     mov.u16         %rh1, %ctaid.x;\n\
 *     mov.u16         %rh2, %ntid.x;\n\
 *     mul.wide.u16    %r2, %rh1, %rh2;\n\
 *     add.u32         %r3, %r1, %r2;\n\
 *     ld.param.u64    %rd1, [__cudaparm__Z8myKernelPi_data];\n\
 *     cvt.s64.s32     %rd2, %r3;\n\
 *     mul.wide.s32    %rd3, %r3, 4;\n\
 *     add.u64         %rd4, %rd1, %rd3;\n\
 *     st.global.s32   [%rd4+0], %r3;\n\
 *     exit;\n\
 *   }\n\
 * ";
 *
 * const char *Entry = "_Z8myKernelPi";
 *
 * int main() {
 *   PollyGPUFunction *Kernel;
 *   PollyGPUContext *Context;
 *   PollyGPUDevicePtr *DevArray;
 *   int *HostData;
 *   int MemSize;
 *
 *   int GridX = 8;
 *   int GridY = 8;
 *
 *   int BlockX = 16;
 *   int BlockY = 16;
 *   int BlockZ = 1;
 *
 *   MemSize = 256*64*sizeof(int);
 *   Context = polly_initContext();
 *   DevArray = polly_allocateMemoryForDevice(MemSize);
 *   Kernel = polly_getKernel(KernelString, KernelName);
 *
 *   void *Params[1];
 *   void *DevPtr = polly_getDevicePtr(DevArray)
 *   Params[0] = &DevPtr;
 *
 *   polly_launchKernel(Kernel, GridX, GridY, BlockX, BlockY, BlockZ, Params);
 *
 *   polly_copyFromDeviceToHost(HostData, DevData, MemSize);
 *   polly_freeKernel(Kernel);
 *   polly_freeDeviceMemory(DevArray);
 *   polly_freeContext(Context);
 * }
 *
 */

typedef struct PollyGPUContextT PollyGPUContext;
typedef struct PollyGPUFunctionT PollyGPUFunction;
typedef struct PollyGPUDevicePtrT PollyGPUDevicePtr;

PollyGPUContext *polly_initContext();
PollyGPUFunction *polly_getKernel(const char *PTXBuffer,
                                  const char *KernelName);
void polly_freeKernel(PollyGPUFunction *Kernel);
void polly_copyFromHostToDevice(void *HostData, PollyGPUDevicePtr *DevData,
                                long MemSize);
void polly_copyFromDeviceToHost(PollyGPUDevicePtr *DevData, void *HostData,
                                long MemSize);
void polly_launchKernel(PollyGPUFunction *Kernel, unsigned int GridDimX,
                        unsigned int GridDimY, unsigned int BlockSizeX,
                        unsigned int BlockSizeY, unsigned int BlockSizeZ,
                        void **Parameters);
void polly_freeDeviceMemory(PollyGPUDevicePtr *Allocation);
void polly_freeContext(PollyGPUContext *Context);
#endif /* GPUJIT_H_ */
