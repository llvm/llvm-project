//===-- AVRSelectionDAGInfo.h - AVR SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AVR subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_SELECTION_DAG_INFO_H
#define LLVM_AVR_SELECTION_DAG_INFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {
/**
 * Holds information about the AVR instruction selection DAG.
 */
class AVRSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
};

} // end namespace llvm

#endif // LLVM_AVR_SELECTION_DAG_INFO_H
