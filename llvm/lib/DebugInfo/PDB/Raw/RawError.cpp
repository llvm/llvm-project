#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
class RawErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override { return "llvm.pdb.raw"; }

  std::string message(int Condition) const override {
    switch (static_cast<raw_error_code>(Condition)) {
    case raw_error_code::unspecified:
      return "An unknown error has occurred.";
    case raw_error_code::feature_unsupported:
      return "The feature is unsupported by the implementation.";
    case raw_error_code::corrupt_file:
      return "The PDB file is corrupt.";
    case raw_error_code::insufficient_buffer:
      return "The buffer is not large enough to read the requested number of "
             "bytes.";
    }
    llvm_unreachable("Unrecognized raw_error_code");
  }
};
} // end anonymous namespace

static ManagedStatic<RawErrorCategory> Category;

char RawError::ID = 0;

RawError::RawError(raw_error_code C) : RawError(C, "") {}

RawError::RawError(const std::string &Context)
    : RawError(raw_error_code::unspecified, Context) {}

RawError::RawError(raw_error_code C, const std::string &Context) : Code(C) {
  ErrMsg = "Native PDB Error: ";
  std::error_code EC = convertToErrorCode();
  if (Code != raw_error_code::unspecified)
    ErrMsg += EC.message() + "  ";
  if (!Context.empty())
    ErrMsg += Context;
}

void RawError::log(raw_ostream &OS) const { OS << ErrMsg << "\n"; }

const std::string &RawError::getErrorMessage() const { return ErrMsg; }

std::error_code RawError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), *Category);
}
