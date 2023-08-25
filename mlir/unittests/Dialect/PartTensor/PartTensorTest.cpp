#include "PartTensorTestBase.h"
using namespace mlir::sparse_tensor;

namespace {
TEST(PartTensor, NewPartTensor) {
  using i64 = int64_t;
  auto const rowSize = 4u;
  auto const partSize = 2u;
  auto dims = std::vector<size_t>{rowSize, rowSize};
  auto parts = std::vector<size_t>{partSize, partSize};
  auto st_coo = std::make_unique<SparseTensorCOO<i64>>(dims);
  llvm::for_each(llvm::seq(0u, rowSize), [&](auto i) {
    st_coo->add({i, i}, 1);
  });
  // auto part_st_coo = PartTensorStorage<i64, i64, i64>::newFromCOO(st_coo,
  // dims);
  const auto elements = st_coo->getElements();
  EXPECT_TRUE(std::all_of(std::begin(elements), std::end(elements),
                          [](auto v) { return v.value == 1; }));
  EXPECT_TRUE(std::size(elements) == rowSize);
  EXPECT_EQ(std::size(dims), st_coo->getRank());
}
} // namespace
