
#include "llvm/Testing/Support/SupportHelpers.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::unittest;

extern const char *TestMainArgv0;

SmallString<128> llvm::unittest::getInputFileDirectory() {
  llvm::SmallString<128> Result = llvm::sys::path::parent_path(TestMainArgv0);
  llvm::sys::fs::make_absolute(Result);
  llvm::sys::path::append(Result, "llvm.srcdir.txt");

  EXPECT_TRUE(llvm::sys::fs::is_regular_file(Result))
      << "Unit test source directory file does not exist.";

  auto File = MemoryBuffer::getFile(Result);

  EXPECT_TRUE(static_cast<bool>(File))
      << "Could not open unit test source directory file.";

  Result.clear();
  Result.append((*File)->getBuffer().trim());
  llvm::sys::path::append(Result, "Inputs");
  llvm::sys::path::native(Result);
  return Result;
}
