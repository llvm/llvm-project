#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::affine;

namespace {

// Core data structures for analyzing loops and memory accesses
struct LoopInfo {
  // Loop bounds and step
  int64_t lowerBound;
  int64_t upperBound;
  int64_t step;
  
  // Memory accesses in this loop
  enum class AccessType {
    Load,
    Store
  };

  struct MemoryAccess {
    Value memref;           // The memref being accessed
    AffineMap accessMap;    // The affine map for the access
    AccessType type;        // Whether it's a load or store
  };
  SmallVector<MemoryAccess> accesses;
};

// Helper class to validate loop structures and memory accesses
class LoopValidator {
public:
  // Check if a memory access is 2D
  static bool is2DAccess(Operation *op) {
    AffineMap map;
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      map = loadOp.getAffineMap();
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      map = storeOp.getAffineMap();
    } else {
      assert(false && "Expected load or store operation");
    }
    return map.getNumResults() == 2;
  }


  // Validate loop band and collect information if valid
  static std::optional<SmallVector<LoopInfo>> validateAndCollectInfo(ArrayRef<AffineForOp> loops) {
    // Check if it's a 2D perfectly nested loop
    if (loops.size() != 2 || !affine::isPerfectlyNested(loops)) {
      return std::nullopt;
    }

    SmallVector<LoopInfo> loopInfos;
    
    // Analyze each loop
    for (const auto &loop : loops) {
      LoopInfo info;
      
      // Get loop bounds and check if they're compile-time constants
      auto lowerMap = const_cast<AffineForOp &>(loop).getLowerBoundMap();
      auto upperMap = const_cast<AffineForOp &>(loop).getUpperBoundMap();
      
      if (!lowerMap.isConstant() || !upperMap.isConstant()) {
        return std::nullopt;
      }
      
      info.lowerBound = lowerMap.getSingleConstantResult();
      info.upperBound = upperMap.getSingleConstantResult();
      info.step = const_cast<AffineForOp &>(loop).getStep().getSExtValue();
      
      // Only collect memory accesses in the innermost loop
      if (loop == loops.back()) {
        bool all2D = true;
        loop->walk([&](Operation *op) {
          if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
            if (!is2DAccess(op)) {
              all2D = false;
              return;
            }
            info.accesses.push_back({loadOp.getMemRef(), loadOp.getAffineMap(), LoopInfo::AccessType::Load});
          } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
            if (!is2DAccess(op)) {
              all2D = false;
              return;
            }
            info.accesses.push_back({storeOp.getMemRef(), storeOp.getAffineMap(), LoopInfo::AccessType::Store});
          }
        });

        // If not all accesses are 2D, return nullopt
        if (!all2D) {
          return std::nullopt;
        }
      }
      
      loopInfos.push_back(info);
    }

    return loopInfos;
  }
};

// Helper function to print loop information
static void printLoopInfo(const SmallVector<LoopInfo> &loopInfos, func::FuncOp funcOp) {
  llvm::errs() << "\n=== Band Information ===\n";
  
  // Print loop structure
  llvm::errs() << "Loop Structure:\n";
  for (size_t i = 0; i < loopInfos.size(); i++) {
    const auto &info = loopInfos[i];
    llvm::errs() << "  Loop " << i << ": [" << info.lowerBound << ", " 
                 << info.upperBound << ") step " << info.step << "\n";
  }
  
  // Print only innermost loop's memory accesses
  llvm::errs() << "\nMemory Accesses in Innermost Loop:\n";
  const auto &innerLoop = loopInfos.back();
  for (const auto &access : innerLoop.accesses) {
    llvm::errs() << "  " << (access.type == LoopInfo::AccessType::Load ? "Load" : "Store") << " from ";
    
    // Print block argument information
    if (auto blockArg = dyn_cast<BlockArgument>(access.memref)) {
      llvm::errs() << "<block argument> of type '" << blockArg.getType() 
                   << "' at index: " << blockArg.getArgNumber()
                   << " (arg" << blockArg.getArgNumber() << ")";
    } else {
      llvm::errs() << access.memref;
    }
    llvm::errs() << "\n";
    llvm::errs() << "    Access Map: " << access.accessMap << "\n";
  }
  llvm::errs() << "================================\n";
}

struct TTLOps : public PassWrapper<TTLOps, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLOps)

  // Default constructor
  TTLOps() = default;

  // Copy constructor - needed for pass cloning
  TTLOps(const TTLOps &other) : PassWrapper<TTLOps, OperationPass<ModuleOp>>(other) {
    // Copy option values
    localMemorySize = other.localMemorySize;
    loadCost = other.loadCost;
    storeCost = other.storeCost;
  }

  // Pass options
  Option<unsigned> localMemorySize{
      *this, "local-memory-size",
      llvm::cl::desc("Size of local memory in KB (default: 32)"),
      llvm::cl::init(32)};
  Option<unsigned> loadCost{
      *this, "load-cost",
      llvm::cl::desc("Cost of a load operation (default: 1)"),
      llvm::cl::init(1)};
  Option<unsigned> storeCost{
      *this, "store-cost",
      llvm::cl::desc("Cost of a store operation (default: 1)"),
      llvm::cl::init(1)};

  StringRef getArgument() const override { return "ttl-ops"; }
  StringRef getDescription() const override { return "TTL operations pass"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Ensure we only have one function in the module
    auto funcOps = module.getOps<func::FuncOp>();
    assert(std::distance(funcOps.begin(), funcOps.end()) == 1 && 
           "Expected exactly one function in the module");
    
    // Find perfect loop nests (bands) in each function
    module->walk([&](func::FuncOp funcOp) {
      std::vector<SmallVector<AffineForOp, 6>> bands;
      mlir::affine::getTileableBands(funcOp, &bands);

      // Analyze each band
      for (const auto &band : bands) {
        // Validate band and collect information
        if (auto loopInfos = LoopValidator::validateAndCollectInfo(band)) {
          printLoopInfo(*loopInfos, funcOp);
        }
      }
    });
  }
};

// Register the pass
void registerTTLOps() {
  PassRegistration<TTLOps>();
}
} // end anonymous namespace

namespace mlir {
std::unique_ptr<Pass> createTTLOpsPass() {
  return std::make_unique<TTLOps>();
}
} // end namespace mlir