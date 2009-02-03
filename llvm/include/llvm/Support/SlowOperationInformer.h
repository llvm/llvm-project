//===- llvm/Support/SlowOperationInformer.h - Keep user informed *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple object which can be used to let the user know what
// is going on when a slow operation is happening, and gives them the ability to
// cancel it.  Potentially slow operations can stack allocate one of these
// objects, and periodically call the "progress" method to update the progress
// bar.  If the operation takes more than 1 second to complete, the progress bar
// is automatically shown and updated.  As such, the slow operation should not
// print stuff to the screen, and should not be confused if an extra line
// appears on the screen (ie, the cursor should be at the start of the line).
//
// If the user presses CTRL-C during the operation, the next invocation of the
// progress method return true indicating that the operation was cancelled.
//
// Because SlowOperationInformers fiddle around with signals, they cannot be
// nested, and interact poorly with threads.  The SIGALRM handler is set back to
// SIGDFL, but the SIGINT signal handler is restored when the
// SlowOperationInformer is destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SLOW_OPERATION_INFORMER_H
#define LLVM_SUPPORT_SLOW_OPERATION_INFORMER_H

#include <string>
#include <cassert>
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class SlowOperationInformer {
    std::string OperationName;
    unsigned LastPrintAmount;

    SlowOperationInformer(const SlowOperationInformer&);   // DO NOT IMPLEMENT
    void operator=(const SlowOperationInformer&);          // DO NOT IMPLEMENT
  public:
    SlowOperationInformer(const std::string &Name);
    ~SlowOperationInformer();

    /// progress - Clients should periodically call this method when they can
    /// handle cancellation.  The Amount variable should indicate how far
    /// along the operation is, given in 1/10ths of a percent (in other words,
    /// Amount should range from 0 to 1000).  If the user cancels the operation,
    /// this returns true, false otherwise.
    bool progress(unsigned Amount);

    /// progress - Same as the method above, but this performs the division for
    /// you, and helps you avoid overflow if you are dealing with largish
    /// numbers.
    bool progress(unsigned Current, unsigned Maximum) {
      assert(Maximum != 0 &&
             "Shouldn't be doing work if there is nothing to do!");
      return progress(Current*uint64_t(1000UL)/Maximum);
    }
  };
} // end namespace llvm

#endif /* SLOW_OPERATION_INFORMER_H */
