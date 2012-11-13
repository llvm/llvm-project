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
 *   PollyGPUContext *Context;
 *   PollyGPUModule *Module;
 *   PollyGPUFunction *Kernel;
 *   PollyGPUDevice *Device;
 *   PollyGPUDevicePtr *PtrDevData;
 *   int *HostData;
 *   PollyGPUEvent *Start;
 *   PollyGPUEvent *Stop;
 *   float *ElapsedTime;
 *   int MemSize;
 *   int BlockWidth = 16;
 *   int BlockHeight = 16;
 *   int GridWidth = 8;
 *   int GridHeight = 8;
 *
 *   MemSize = 256*64*sizeof(int);
 *   polly_initDevice(&Context, &Device);
 *   polly_getPTXModule(KernelString, &Module);
 *   polly_getPTXKernelEntry(Entry, Module, &Kernel);
 *   polly_allocateMemoryForHostAndDevice(&HostData, &DevData, MemSize);
 *   polly_setKernelParameters(Kernel, BlockWidth, BlockHeight, DevData);
 *   polly_startTimerByCudaEvent(&Start, &Stop);
 *   polly_launchKernel(Kernel, GridWidth, GridHeight);
 *   polly_copyFromDeviceToHost(HostData, DevData, MemSize);
 *   polly_stopTimerByCudaEvent(Start, Stop, ElapsedTime);
 *   polly_cleanupGPGPUResources(HostData, DevData, Module, Context, Kernel);
 * }
 *
 */

typedef struct PollyGPUContextT PollyGPUContext;
typedef struct PollyGPUModuleT PollyGPUModule;
typedef struct PollyGPUFunctionT PollyGPUFunction;
typedef struct PollyGPUDeviceT PollyGPUDevice;
typedef struct PollyGPUDevicePtrT PollyGPUDevicePtr;
typedef struct PollyGPUEventT PollyGPUEvent;

void polly_initDevice(PollyGPUContext **Context, PollyGPUDevice **Device);
void polly_getPTXModule(void *PTXBuffer, PollyGPUModule **Module);
void polly_getPTXKernelEntry(const char *KernelName, PollyGPUModule *Module,
                             PollyGPUFunction **Kernel);
void polly_startTimerByCudaEvent(PollyGPUEvent **Start, PollyGPUEvent **Stop);
void polly_stopTimerByCudaEvent(PollyGPUEvent *Start, PollyGPUEvent *Stop,
                                float *ElapsedTimes);
void polly_copyFromHostToDevice(PollyGPUDevicePtr *DevData, void *HostData,
                                int MemSize);
void polly_copyFromDeviceToHost(void *HostData, PollyGPUDevicePtr *DevData,
                                int MemSize);
void polly_allocateMemoryForHostAndDevice(void **HostData,
                                          PollyGPUDevicePtr **DevData,
                                          int MemSize);
void polly_setKernelParameters(PollyGPUFunction *Kernel, int BlockWidth,
                               int BlockHeight, PollyGPUDevicePtr *DevData);
void polly_launchKernel(PollyGPUFunction *Kernel, int GridWidth,
                        int GridHeight);
void polly_cleanupGPGPUResources(void *HostData, PollyGPUDevicePtr *DevData,
                                 PollyGPUModule *Module,
                                 PollyGPUContext *Context,
                                 PollyGPUFunction *Kernel);
#endif /* GPUJIT_H_ */
