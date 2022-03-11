//===- TestOpaqueLoc.cpp - Pass to test opaque locations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Pass that changes locations to opaque locations for each operation.
/// It also takes all operations that are not function operations or
/// terminators and clones them with opaque locations which store the initial
/// locations.
struct TestOpaqueLoc
    : public PassWrapper<TestOpaqueLoc, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-opaque-loc"; }
  StringRef getDescription() const final {
    return "Changes all leaf locations to opaque locations";
  }

  /// A simple structure which is used for testing as an underlying location in
  /// OpaqueLoc.
  struct MyLocation {
    MyLocation() = default;
    MyLocation(int id) : id(id) {}
    int getId() { return id; }

    int id{42};
  };

  void runOnOperation() override {
    std::vector<std::unique_ptr<MyLocation>> myLocs;
    int lastIt = 0;

    getOperation().getBody()->walk([&](Operation *op) {
      myLocs.push_back(std::make_unique<MyLocation>(lastIt++));

      Location loc = op->getLoc();

      /// Set opaque location without fallback location to test the
      /// corresponding get method.
      op->setLoc(
          OpaqueLoc::get<MyLocation *>(myLocs.back().get(), &getContext()));

      if (isa<ModuleOp>(op->getParentOp()) ||
          op->hasTrait<OpTrait::IsTerminator>())
        return;

      OpBuilder builder(op);

      /// Add the same operation but with fallback location to test the
      /// corresponding get method and serialization.
      Operation *opCloned1 = builder.clone(*op);
      opCloned1->setLoc(OpaqueLoc::get<MyLocation *>(myLocs.back().get(), loc));

      /// Add the same operation but with void* instead of MyLocation* to test
      /// getUnderlyingLocationOrNull method.
      Operation *opCloned2 = builder.clone(*op);
      opCloned2->setLoc(OpaqueLoc::get<void *>(nullptr, loc));
    });

    ScopedDiagnosticHandler diagHandler(&getContext(), [](Diagnostic &diag) {
      auto &os = llvm::outs();
      if (diag.getLocation().isa<OpaqueLoc>()) {
        MyLocation *loc = OpaqueLoc::getUnderlyingLocationOrNull<MyLocation *>(
            diag.getLocation());
        if (loc)
          os << "MyLocation: " << loc->id;
        else
          os << "nullptr";
      }
      os << ": " << diag << '\n';
      os.flush();
    });

    getOperation().walk([&](Operation *op) { op->emitOpError(); });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestOpaqueLoc() { PassRegistration<TestOpaqueLoc>(); }
} // namespace test
} // namespace mlir
