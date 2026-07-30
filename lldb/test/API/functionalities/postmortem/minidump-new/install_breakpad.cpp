#include "client/linux/handler/exception_handler.h"

static bool dumpCallback(const google_breakpad::MinidumpDescriptor &descriptor,
                         void *context, bool succeeded) {
  return succeeded;
}

google_breakpad::ExceptionHandler *eh;

void InstallBreakpad() {
  google_breakpad::MinidumpDescriptor descriptor("/tmp");
  eh = new google_breakpad::ExceptionHandler(descriptor, NULL, dumpCallback,
                                             NULL, true, -1);
}

void WriteMinidump() { eh->WriteMinidump(); }
