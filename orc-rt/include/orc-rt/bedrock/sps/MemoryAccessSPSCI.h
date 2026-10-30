//===-- MemoryAccessSPSCI.h -- MemoryAccess SPS CI --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for MemoryAccess.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_MEMORYACCESSSPSCI_H
#define ORC_RT_BEDROCK_SPS_MEMORYACCESSSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_write_uint8s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_write_uint16s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_write_uint32s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_write_uint64s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_write_pointers)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_write_buffers)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_uint8s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_uint16s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_uint32s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_uint64s)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_pointers)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_buffers)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_mem_read_strings)

namespace orc_rt::sps_ci {

/// Add the MemoryAccess SPS interface to the controller interface.
Error addMemoryAccess(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_MEMORYACCESSSPSCI_H
