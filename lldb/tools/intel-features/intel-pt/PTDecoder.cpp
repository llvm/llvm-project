//===-- PTDecoder.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "PTDecoder.h"
#include "Decoder.h"

using namespace ptdecoder;
using namespace ptdecoder_private;

// PTInstruction class member functions definitions
PTInstruction::PTInstruction() : m_opaque_sp() {}

PTInstruction::PTInstruction(const PTInstruction &insn)
    : m_opaque_sp(insn.m_opaque_sp) {}

PTInstruction::PTInstruction(
    const std::shared_ptr<ptdecoder_private::Instruction> &ptr)
    : m_opaque_sp(ptr) {}

PTInstruction::~PTInstruction() {}

uint64_t PTInstruction::GetInsnAddress() const {
  return (m_opaque_sp ? m_opaque_sp->GetInsnAddress() : 0);
}

size_t PTInstruction::GetRawBytes(void *buf, size_t size) const {
  return (m_opaque_sp ? m_opaque_sp->GetRawBytes(buf, size) : 0);
}

std::string PTInstruction::GetError() const {
  return (m_opaque_sp ? m_opaque_sp->GetError() : "null pointer");
}

bool PTInstruction::GetSpeculative() const {
  return (m_opaque_sp ? m_opaque_sp->GetSpeculative() : 0);
}

// PTInstructionList class member functions definitions
PTInstructionList::PTInstructionList() : m_opaque_sp() {}

PTInstructionList::PTInstructionList(const PTInstructionList &insn_list)
    : m_opaque_sp(insn_list.m_opaque_sp) {}

PTInstructionList::~PTInstructionList() {}

size_t PTInstructionList::GetSize() const {
  return (m_opaque_sp ? m_opaque_sp->GetSize() : 0);
}

PTInstruction PTInstructionList::GetInstructionAtIndex(uint32_t idx) {
  if (m_opaque_sp)
    return PTInstruction(std::shared_ptr<ptdecoder_private::Instruction>(
        new Instruction(m_opaque_sp->GetInstructionAtIndex(idx))));

  return PTInstruction(std::shared_ptr<ptdecoder_private::Instruction>(
      new Instruction("invalid instruction")));
}

void PTInstructionList::SetSP(
    const std::shared_ptr<ptdecoder_private::InstructionList> &ptr) {
  m_opaque_sp = ptr;
}
void PTInstructionList::Clear() {
  if (!m_opaque_sp)
    return;
  m_opaque_sp.reset();
}

// PTTraceOptions class member functions definitions
PTTraceOptions::PTTraceOptions() : m_opaque_sp() {}

PTTraceOptions::PTTraceOptions(const PTTraceOptions &options)
    : m_opaque_sp(options.m_opaque_sp) {}

PTTraceOptions::~PTTraceOptions() {}

lldb::TraceType PTTraceOptions::GetType() const {
  return (m_opaque_sp ? m_opaque_sp->getType()
                      : lldb::TraceType::eTraceTypeNone);
}

uint64_t PTTraceOptions::GetTraceBufferSize() const {
  return (m_opaque_sp ? m_opaque_sp->getTraceBufferSize() : 0);
}

uint64_t PTTraceOptions::GetMetaDataBufferSize() const {
  return (m_opaque_sp ? m_opaque_sp->getMetaDataBufferSize() : 0);
}

lldb::SBStructuredData PTTraceOptions::GetTraceParams(lldb::SBError &error) {
  if (!m_opaque_sp)
    error.SetErrorString("null pointer");
  return (m_opaque_sp ? m_opaque_sp->getTraceParams(error)
                      : lldb::SBStructuredData());
}

void PTTraceOptions::SetSP(
    const std::shared_ptr<ptdecoder_private::TraceOptions> &ptr) {
  m_opaque_sp = ptr;
}

// PTDecoder class member functions definitions
PTDecoder::PTDecoder(lldb::SBDebugger &sbdebugger)
    : m_opaque_sp(new ptdecoder_private::Decoder(sbdebugger)) {}

PTDecoder::PTDecoder(const PTDecoder &ptdecoder)
    : m_opaque_sp(ptdecoder.m_opaque_sp) {}

PTDecoder::~PTDecoder() {}

void PTDecoder::StartProcessorTrace(lldb::SBProcess &sbprocess,
                                    lldb::SBTraceOptions &sbtraceoptions,
                                    lldb::SBError &sberror) {
  if (m_opaque_sp == nullptr) {
    sberror.SetErrorStringWithFormat("invalid PTDecoder instance");
    return;
  }

  m_opaque_sp->StartProcessorTrace(sbprocess, sbtraceoptions, sberror);
}

void PTDecoder::StopProcessorTrace(lldb::SBProcess &sbprocess,
                                   lldb::SBError &sberror, lldb::tid_t tid) {
  if (m_opaque_sp == nullptr) {
    sberror.SetErrorStringWithFormat("invalid PTDecoder instance");
    return;
  }

  m_opaque_sp->StopProcessorTrace(sbprocess, sberror, tid);
}

void PTDecoder::GetInstructionLogAtOffset(lldb::SBProcess &sbprocess,
                                          lldb::tid_t tid, uint32_t offset,
                                          uint32_t count,
                                          PTInstructionList &result_list,
                                          lldb::SBError &sberror) {
  if (m_opaque_sp == nullptr) {
    sberror.SetErrorStringWithFormat("invalid PTDecoder instance");
    return;
  }

  std::shared_ptr<ptdecoder_private::InstructionList> insn_list_ptr(
      new InstructionList());
  m_opaque_sp->GetInstructionLogAtOffset(sbprocess, tid, offset, count,
                                         *insn_list_ptr, sberror);
  if (!sberror.Success())
    return;

  result_list.SetSP(insn_list_ptr);
}

void PTDecoder::GetProcessorTraceInfo(lldb::SBProcess &sbprocess,
                                      lldb::tid_t tid, PTTraceOptions &options,
                                      lldb::SBError &sberror) {
  if (m_opaque_sp == nullptr) {
    sberror.SetErrorStringWithFormat("invalid PTDecoder instance");
    return;
  }

  std::shared_ptr<ptdecoder_private::TraceOptions> trace_options_ptr(
      new TraceOptions());
  m_opaque_sp->GetProcessorTraceInfo(sbprocess, tid, *trace_options_ptr,
                                     sberror);
  if (!sberror.Success())
    return;

  options.SetSP(trace_options_ptr);
}
