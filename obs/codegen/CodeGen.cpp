
#include "MLIRGen.h"

#include "AST.h"
#include "Dialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <functional>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>

#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

#include "CodeGen.h"

namespace mlir {
namespace obs {

bool MLIRGenImpl::VisitFunctionDecl(clang::FunctionDecl *funcDecl) {
  llvm::outs() << "VisitFunctionDecl: ";
  funcDecl->getDeclName().dump();
  llvm::outs() << "\n";
  return false;
}

} // namespace obs
} // namespace mlir
