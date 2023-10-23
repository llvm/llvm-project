#include "PartTensorTestBase.h"
#include "mlir/ExecutionEngine/PartTensor/Storage.h"
using namespace mlir::sparse_tensor;

using index_t = uint64_t;

auto getColumnPartitions2d(size_t totalColumns, size_t numPartitions) {
  std::vector<index_t> partitionPlan;
  auto partitionSize = totalColumns / numPartitions;
  for (auto i : llvm::seq(index_t(0), index_t(numPartitions))) {
    auto leftx = i * partitionSize;
    auto lefty = 0;
    auto rightx = (i + 1) * partitionSize;
    auto righty = totalColumns;
    partitionPlan.push_back(index_t(leftx));
    partitionPlan.push_back(index_t(lefty));
    partitionPlan.push_back(index_t(rightx));
    partitionPlan.push_back(index_t(righty));
  }
  return partitionPlan;
}

namespace {
TEST(PartTensor, NewPartTensor) {
  auto const rowSize = 4u;
  auto const partSize = 2u;
  auto dims = std::vector<size_t>{rowSize, rowSize};
  auto stCoo = std::make_unique<SparseTensorCOO<float>>(dims);
  auto partitionPlan = getColumnPartitions2d(rowSize, partSize);
  llvm::for_each(llvm::seq(0u, rowSize), [&](auto i) {
    stCoo->add({i, i}, 1.0);
  });
  const auto elements = stCoo->getElements();
  EXPECT_TRUE(std::all_of(std::begin(elements), std::end(elements),
                          [](auto v) { return v.value == 1.0; }));
  EXPECT_TRUE(std::size(elements) == rowSize);
  EXPECT_EQ(std::size(dims), stCoo->getRank());
  auto pt = PartTensorStorage<index_t, index_t, float>::newFromCOO(
      std::size(partitionPlan), partitionPlan.data(), 2, dims.data(),
      stCoo.get());
  {
    auto &parts = pt->getParts();
    for (auto p : llvm::seq(0ul, std::size(parts))) {
      std::cout << "Part:\n";
      for (auto i : parts[p]->getElements()) {
        std::cout << i.coords[0] << ", " << i.coords[1] << ", " << i.value
                  << "\n";
      }
      std::cout << "----\n";
    }
  }
}
} // namespace
