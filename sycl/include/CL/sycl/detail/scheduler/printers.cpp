//==----------- printers.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/scheduler/commands.h>
#include <CL/sycl/device.hpp>
#include <CL/sycl/queue.hpp>

#include <ostream>

namespace cl {
namespace sycl {
namespace simple_scheduler {

static std::string accessMode2String(cl::sycl::access::mode Type) {
  switch (Type) {
  case access::mode::write:
    return "write";
  case access::mode::read:
    return "read";
  case access::mode::read_write:
    return "read_write";
  default:
    return "unhandled";
  }
}

static std::string
getDeviceTypeString(const cl::sycl::device &Device,
                    access::target Target = access::target::global_buffer) {
  if (access::target::host_buffer == Target) {
    return "User host.";
  }
  if (Device.is_cpu()) {
    return "CPU";
  }
  if (Device.is_gpu()) {
    return "GPU";
  }
  if (Device.is_accelerator()) {
    return "ACC";
  }
  if (Device.is_host()) {
    return "HOST";
  }
  return "";
}

static std::string
getColor(const cl::sycl::device &Device,
         access::target Target = access::target::global_buffer) {
  if (access::target::host_buffer == Target) {
    return "#FFDEAD"; // navajowhite1
  }
  if (Device.is_cpu()) {
    return "#00BFFF"; // deepskyblue1
  }
  if (Device.is_gpu()) {
    return "#00FF7F"; // green
  }
  if (Device.is_accelerator()) {
    return "#FF0000"; // red
  }
  if (Device.is_host()) {
    return "#FFBBFF"; // plum1
  }
  return "";
}

template <typename KernelType, int Dimensions, typename RangeType,
          typename KernelArgType, bool SingleTask>
void ExecuteKernelCommand<KernelType, Dimensions, RangeType, KernelArgType,
                          SingleTask>::printDot(std::ostream &Stream) const {
  const std::string CommandColor = getColor(m_Queue->get_device());

  Stream << "\"" << this << "\" [style=filled, label=\"";

  Stream << "ID = " << getID() << " ; ";
  Stream << "RUN_KERNEL "
         << "\\n"
         << m_KernelName << " ON ";
  Stream << getDeviceTypeString(m_Queue->get_device()) << "\\n";

  Stream << "\", fillcolor=\"" << CommandColor << "\"];" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Buf = Dep.second;
    Stream << "  \"" << this << "\" -> \"" << Dep.first << "\" [ label=\"";
    Stream << accessMode2String(Buf->getAccessModeType()) << "\" ];";
    Stream << std::endl;
  }
}

template <typename KernelType, int Dimensions, typename RangeType,
          typename KernelArgType, bool SingleTask>
void ExecuteKernelCommand<KernelType, Dimensions, RangeType, KernelArgType,
                          SingleTask>::print(std::ostream &Stream) const {
  Stream << "ID = " << getID() << " ; ";
  Stream << "RUN_KERNEL " << m_KernelName << " ON ";
  Stream << getDeviceTypeString(m_Queue->get_device()) << std::endl;
  Stream << "    Dependency:" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Command = Dep.first;
    const auto &Buf = Dep.second;
    Stream << "        Dep on buf " << Buf->getUniqID() << " ";
    Stream << accessMode2String(Buf->getAccessModeType());
    Stream << " from Command ID = " << Command->getID() << std::endl;
  }
}

template <typename T, int Dim>
void FillCommand<T, Dim>::printDot(std::ostream &Stream) const {
  const std::string CommandColor = getColor(m_Queue->get_device());

  Stream << "\"" << this << "\" [style=filled, label=\"";

  Stream << "ID = " << getID() << " ; ";
  Stream << "Fill "
         << "\\n"
         << " Buf : " << m_Buf->getUniqID() << " ON ";
  Stream << getDeviceTypeString(m_Queue->get_device()) << "\\n";

  Stream << "\", fillcolor=\"" << CommandColor << "\"];" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Buf = Dep.second;
    Stream << "  \"" << this << "\" -> \"" << Dep.first << "\" [ label=\"";
    Stream << accessMode2String(Buf->getAccessModeType()) << "\" ];";
    Stream << std::endl;
  }
}

template <typename T, int Dim>
void FillCommand<T, Dim>::print(std::ostream &Stream) const {
  Stream << "ID = " << getID() << " ; ";
  Stream << "Fill "
         << " Buf : " << m_Buf->getUniqID() << " ON ";
  Stream << getDeviceTypeString(m_Queue->get_device()) << std::endl;
  Stream << "    Dependency:" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Command = Dep.first;
    const auto &Buf = Dep.second;
    Stream << "        Dep on buf " << Buf->getUniqID() << " ";
    Stream << accessMode2String(Buf->getAccessModeType());
    Stream << " from Command ID = " << Command->getID() << std::endl;
  }
}

template <int DimSrc, int DimDest>
void CopyCommand<DimSrc, DimDest>::printDot(std::ostream &Stream) const {
  const std::string CommandColor = getColor(m_Queue->get_device());

  Stream << "\"" << this << "\" [style=filled, label=\"";

  Stream << "ID = " << getID() << " ; ";
  Stream << "Copy "
         << "\\n"
         << " Buf : " << m_BufSrc->getUniqID() << " ON ";
  Stream << getDeviceTypeString(m_Queue->get_device()) << "\\n";
  Stream << " To Buf : " << m_BufDest->getUniqID();

  Stream << "\", fillcolor=\"" << CommandColor << "\"];" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Buf = Dep.second;
    Stream << "  \"" << this << "\" -> \"" << Dep.first << "\" [ label=\"";
    Stream << accessMode2String(Buf->getAccessModeType()) << "\" ];";
    Stream << std::endl;
  }
}

template <int DimSrc, int DimDest>
void CopyCommand<DimSrc, DimDest>::print(std::ostream &Stream) const {
  Stream << "ID = " << getID() << " ; ";
  Stream << "Copy "
         << " Buf : " << m_BufSrc->getUniqID() << " ON ";
  Stream << getDeviceTypeString(m_Queue->get_device()) << std::endl;
  Stream << " Buf : " << m_BufDest->getUniqID();
  Stream << "    Dependency:" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Command = Dep.first;
    const auto &Buf = Dep.second;
    Stream << "        Dep on buf " << Buf->getUniqID() << " ";
    Stream << accessMode2String(Buf->getAccessModeType());
    Stream << " from Command ID = " << Command->getID() << std::endl;
  }
}

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
