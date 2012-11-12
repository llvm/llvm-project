//===- lib/Support/ErrorHandling.cpp - Callbacks for errors ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an API used to indicate fatal error conditions.  Non-fatal
// errors (most of them) should be handled through LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/config.h"
#include <cassert>
#include <cstdlib>

#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif
#if defined(_MSC_VER)
# include <io.h>
# include <fcntl.h>
#endif

using namespace llvm;

static fatal_error_handler_t ErrorHandler = 0;
static void *ErrorHandlerUserData = 0;

void llvm::install_fatal_error_handler(fatal_error_handler_t handler,
                                       void *user_data) {
  assert(!llvm_is_multithreaded() &&
         "Cannot register error handlers after starting multithreaded mode!\n");
  assert(!ErrorHandler && "Error handler already registered!\n");
  ErrorHandler = handler;
  ErrorHandlerUserData = user_data;
}

void llvm::remove_fatal_error_handler() {
  ErrorHandler = 0;
}

void llvm::report_fatal_error(const char *Reason) {
  report_fatal_error(Twine(Reason));
}

void llvm::report_fatal_error(const std::string &Reason) {
  report_fatal_error(Twine(Reason));
}

void llvm::report_fatal_error(StringRef Reason) {
  report_fatal_error(Twine(Reason));
}

void llvm::report_fatal_error(const Twine &Reason) {
  if (ErrorHandler) {
    ErrorHandler(ErrorHandlerUserData, Reason.str());
  } else {
    // Blast the result out to stderr.  We don't try hard to make sure this
    // succeeds (e.g. handling EINTR) and we can't use errs() here because
    // raw ostreams can call report_fatal_error.
    SmallVector<char, 64> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << "LLVM ERROR: " << Reason << "\n";
    StringRef MessageStr = OS.str();
    ssize_t written = ::write(2, MessageStr.data(), MessageStr.size());
    (void)written; // If something went wrong, we deliberately just give up.
  }

  // If we reached here, we are failing ungracefully. Run the interrupt handlers
  // to make sure any special cleanups get done, in particular that we remove
  // files registered with RemoveFileOnSignal.
  sys::RunInterruptHandlers();

  exit(1);
}

void llvm::llvm_unreachable_internal(const char *msg, const char *file,
                                     unsigned line) {
  // This code intentionally doesn't call the ErrorHandler callback, because
  // llvm_unreachable is intended to be used to indicate "impossible"
  // situations, and not legitimate runtime errors.
  if (msg)
    dbgs() << msg << "\n";
  dbgs() << "UNREACHABLE executed";
  if (file)
    dbgs() << " at " << file << ":" << line;
  dbgs() << "!\n";
  abort();
}
