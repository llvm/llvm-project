//==------------ sycl.hpp - SYCL standard header file ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/image.hpp>
#include <CL/sycl/intel/sub_group.hpp>
#include <CL/sycl/item.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/macro.hpp>
#include <CL/sycl/math.hpp>
#include <CL/sycl/multi_ptr.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/program.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>

// Do not include RT only function implementations for device code as it leads
// to problem. Should be finally fixed when we introduce library.
#ifndef __SYCL_DEVICE_ONLY__
// The following files are supposed to be included after all SYCL classes
// processed.
#include <CL/sycl/detail/scheduler/commands.cpp>
#include <CL/sycl/detail/scheduler/printers.cpp>
#include <CL/sycl/detail/scheduler/scheduler.cpp>
#endif //__SYCL_DEVICE_ONLY__
