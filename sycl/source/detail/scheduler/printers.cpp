//==----------- printers.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/commands.h>
#include <CL/sycl/detail/scheduler/printers.cpp>
#include <CL/sycl/device.hpp>

#include <ostream>

namespace cl {
namespace sycl {
namespace simple_scheduler {

void MemMoveCommand::printDot(std::ostream &Stream) const {
  cl::sycl::device DstDevice = m_Queue->get_device();
  cl::sycl::device SrcDevice = m_SrcQueue->get_device();
  const std::string ToDevColor = getColor(DstDevice, m_Buf->getTargetType());
  std::string FromDevColor = getColor(SrcDevice);

  Stream << "\"" << this << "\" [style=filled, gradientangle=90, label=\"";

  Stream << "ID = " << getID() << " ; ";
  Stream << "MOVE TO " << getDeviceTypeString(DstDevice) << "\\n";
  Stream << "  Buf : " << m_Buf->getUniqID();
  Stream << "  Access : " << accessMode2String(m_AccessMode) << "\\n";

  Stream << "\", fillcolor=\"" << FromDevColor;
  Stream << ";0.5:" << ToDevColor << "\"];" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Buf = Dep.second;
    Stream << "\"" << this << "\" -> \"" << Dep.first << "\" [ label=\"";
    Stream << accessMode2String(Buf->getAccessModeType()) << "\" ];";
    Stream << std::endl;
  }
}

void MemMoveCommand::print(std::ostream &Stream) const {
  Stream << "ID = " << getID() << " ; ";
  Stream << "MOVE TO " << getDeviceTypeString(m_Queue->get_device())
         << std::endl;
  Stream << "  Buf : " << m_Buf->getUniqID();
  Stream << "  Access : " << accessMode2String(m_AccessMode) << std::endl;
  Stream << "    Dependency:" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Command = Dep.first;
    const auto &Buf = Dep.second;
    Stream << "        Dep on buf " << Buf->getUniqID() << " ";
    Stream << accessMode2String(Buf->getAccessModeType());
    Stream << " from Command ID = " << Command->getID() << std::endl;
  }
}

void AllocaCommand::printDot(std::ostream &Stream) const {

  const std::string CommandColor = getColor(m_Queue->get_device());

  Stream << "\"" << this << "\" [style=filled, label=\"";

  Stream << "ID = " << getID() << " ; ";
  Stream << "ALLOCA ON " << getDeviceTypeString(m_Queue->get_device()) << "\\n";
  Stream << " Buf : " << m_Buf->getUniqID();
  Stream << " Access : " << accessMode2String(m_AccessMode) << "\\n";

  Stream << "\", fillcolor=\"" << CommandColor << "\"];" << std::endl;

  for (const auto &Dep : m_Deps) {
    const auto &Buf = Dep.second;
    Stream << "  \"" << this << "\" -> \"" << Dep.first << "\" [ label=\"";
    Stream << accessMode2String(Buf->getAccessModeType()) << "\" ];";
    Stream << std::endl;
  }
}

void AllocaCommand::print(std::ostream &Stream) const {
  Stream << "ID = " << getID() << " ; ";
  Stream << "ALLOCA ON " << getDeviceTypeString(m_Queue->get_device())
         << std::endl;
  Stream << "  Buf : " << m_Buf->getUniqID();
  Stream << "  Access : " << accessMode2String(m_AccessMode) << std::endl;
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
