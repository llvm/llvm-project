#ifndef TOOLS_LLVM_DWP_DWPERROR
#define TOOLS_LLVM_DWP_DWPERROR

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

namespace llvm {
class DWPError : public ErrorInfo<DWPError> {
public:
  DWPError(std::string Info) : Info(std::move(Info)) {}
  void log(raw_ostream &OS) const override { OS << Info; }
  std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
  static char ID;

private:
  std::string Info;
};
}

#endif
