//===-- llvm/Support/Signposts.h - Interval debug annotations ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Some OS's provide profilers that allow applications to provide custom
/// annotations to the profiler. For example, on Xcode 10 and later 'signposts'
/// can be emitted by the application and these will be rendered to the Points
/// of Interest track on the instruments timeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIGNPOSTS_H
#define LLVM_SUPPORT_SIGNPOSTS_H

namespace llvm {
class SignpostEmitterImpl;
class Timer;

/// Manages the emission of signposts into the recording method supported by
/// the OS.
class SignpostEmitter {
  SignpostEmitterImpl *Impl;

public:
  SignpostEmitter();
  ~SignpostEmitter();

  bool isEnabled() const;

  /// Begin a signposted interval for the given timer.
  void startTimerInterval(Timer *T);
  /// End a signposted interval for the given timer.
  void endTimerInterval(Timer *T);
};

} // end namespace llvm

#endif // ifndef LLVM_SUPPORT_SIGNPOSTS_H
