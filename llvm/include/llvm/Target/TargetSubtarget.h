//==-- llvm/Target/TargetSubtarget.h - Target Information --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSUBTARGET_H
#define LLVM_TARGET_TARGETSUBTARGET_H

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// TargetSubtarget - Generic base class for all target subtargets.  All
/// Target-specific options that control code generation and printing should
/// be exposed through a TargetSubtarget-derived class.
///
class TargetSubtarget {
  TargetSubtarget(const TargetSubtarget&);   // DO NOT IMPLEMENT
  void operator=(const TargetSubtarget&);  // DO NOT IMPLEMENT
protected: // Can only create subclasses...
  TargetSubtarget();
public:
  virtual ~TargetSubtarget();

  /// getSpecialAddressLatency - For targets where it is beneficial to
  /// backschedule instructions that compute addresses, return a value
  /// indicating the number of scheduling cycles of backscheduling that
  /// should be attempted.
  virtual unsigned getSpecialAddressLatency() const { return 0; }
};

} // End llvm namespace

#endif
