#include "polly/Support/ISLTools.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace isl {
static bool operator==(const isl::basic_set &A, const isl::basic_set &B) {
  return A.is_equal(B);
}
} // namespace isl

TEST(Support, isl_iterator) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> RawCtx(isl_ctx_alloc(),
                                                           &isl_ctx_free);
  isl::ctx Ctx(RawCtx.get());

  isl::basic_set A(
      Ctx, "{ [x, y] : 0 <= x <= 5 and y >= 0 and x > 0 and 0 < y <= 5 }");
  isl::basic_set B(
      Ctx, "{ [x, y] : 0 <= x <= 5 and y >= 0 and x <= 4 and y <= 3 + x }");
  isl::set S = A.unite(B);

  ASSERT_EQ(S.n_basic_set(), 2);
  std::vector<isl::basic_set> Sets;
  for (auto BS : S.get_basic_set_list())
    Sets.push_back(BS);
  EXPECT_THAT(Sets, testing::UnorderedElementsAre(A, B));
}
