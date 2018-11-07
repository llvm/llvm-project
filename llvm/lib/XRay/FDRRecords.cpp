//===- FDRRecords.cpp -  XRay Flight Data Recorder Mode Records -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Define types and operations on these types that represent the different kinds
// of records we encounter in XRay flight data recorder mode traces.
//
//===----------------------------------------------------------------------===//
#include "llvm/XRay/FDRRecords.h"

namespace llvm {
namespace xray {

Error BufferExtents::apply(RecordVisitor &V) { return V.visit(*this); }
Error WallclockRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error NewCPUIDRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error TSCWrapRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error CustomEventRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error CallArgRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error PIDRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error NewBufferRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error EndBufferRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error FunctionRecord::apply(RecordVisitor &V) { return V.visit(*this); }
Error CustomEventRecordV5::apply(RecordVisitor &V) { return V.visit(*this); }
Error TypedEventRecord::apply(RecordVisitor &V) { return V.visit(*this); }

} // namespace xray
} // namespace llvm
