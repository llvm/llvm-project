#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <OffloadAPI.h>
#include <math.h>

llvm::StringRef DeviceBinsDirectory = DEVICE_CODE_PATH;

int main() { llvm::errs() << sin(0.0) << "\n"; }
