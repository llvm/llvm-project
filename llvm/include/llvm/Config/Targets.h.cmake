/*===------- llvm/Config/Targets.h - LLVM target checks -----------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file enables clients to know whether specific targets are enabled. */

#ifndef LLVM_CONFIG_TARGETS_H
#define LLVM_CONFIG_TARGETS_H

/* Define if the AArch64 target is built in */
#cmakedefine01 LLVM_HAS_AARCH64_TARGET

/* Define if the AMDGPU target is built in */
#cmakedefine01 LLVM_HAS_AMDGPU_TARGET

/* Define if the ARC target is built in */
#cmakedefine01 LLVM_HAS_ARC_TARGET

/* Define if the ARM target is built in */
#cmakedefine01 LLVM_HAS_ARM_TARGET

/* Define if the AVR target is built in */
#cmakedefine01 LLVM_HAS_AVR_TARGET

/* Define if the BPF target is built in */
#cmakedefine01 LLVM_HAS_BPF_TARGET

/* Define if the CSKY target is built in */
#cmakedefine01 LLVM_HAS_CSKY_TARGET

/* Define if the DirectX target is built in */
#cmakedefine01 LLVM_HAS_DIRECTX_TARGET

/* Define if the Hexagon target is built in */
#cmakedefine01 LLVM_HAS_HEXAGON_TARGET

/* Define if the Lanai target is built in */
#cmakedefine01 LLVM_HAS_LANAI_TARGET

/* Define if the LoongArch target is built in */
#cmakedefine01 LLVM_HAS_LOONGARCH_TARGET

/* Define if the M68k target is built in */
#cmakedefine01 LLVM_HAS_M68K_TARGET

/* Define if the Mips target is built in */
#cmakedefine01 LLVM_HAS_MIPS_TARGET

/* Define if the MSP430 target is built in */
#cmakedefine01 LLVM_HAS_MSP430_TARGET

/* Define if the NVPTX target is built in */
#cmakedefine01 LLVM_HAS_NVPTX_TARGET

/* Define if the PowerPC target is built in */
#cmakedefine01 LLVM_HAS_POWERPC_TARGET

/* Define if the RISCV target is built in */
#cmakedefine01 LLVM_HAS_RISCV_TARGET

/* Define if the Sparc target is built in */
#cmakedefine01 LLVM_HAS_SPARC_TARGET

/* Define if the SPIRV target is built in */
#cmakedefine01 LLVM_HAS_SPIRV_TARGET

/* Define if the SystemZ target is built in */
#cmakedefine01 LLVM_HAS_SYSTEMZ_TARGET

/* Define if the VE target is built in */
#cmakedefine01 LLVM_HAS_VE_TARGET

/* Define if the WebAssembly target is built in */
#cmakedefine01 LLVM_HAS_WEBASSEMBLY_TARGET

/* Define if the X86 target is built in */
#cmakedefine01 LLVM_HAS_X86_TARGET

/* Define if the XCore target is built in */
#cmakedefine01 LLVM_HAS_XCORE_TARGET

/* Define if the Xtensa target is built in */
#cmakedefine01 LLVM_HAS_XTENSA_TARGET

#endif
