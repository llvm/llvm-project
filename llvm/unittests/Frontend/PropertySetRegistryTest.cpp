#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm::offloading::sycl;
using namespace llvm;

void checkEquality(const PropertySetRegistry &PSR1,
                   const PropertySetRegistry &PSR2) {
  ASSERT_EQ(PSR1.getPropSets().size(), PSR2.getPropSets().size());
  for (const auto &[Category1, PropSet1] : PSR1.getPropSets()) {
    auto It = PSR2.getPropSets().find(Category1);
    ASSERT_TRUE(It != PSR2.getPropSets().end());
    const auto &[Category2, PropSet2] = *It;
    ASSERT_EQ(PropSet1.size(), PropSet2.size());
    for (auto It1 = PropSet1.begin(), It2 = PropSet2.begin(),
              E = PropSet1.end();
         It1 != E; ++It1, ++It2) {
      const auto &[PropName1, PropValue1] = *It1;
      const auto &[PropName2, PropValue2] = *It2;
      ASSERT_EQ(PropName1, PropName2);
      ASSERT_EQ(PropValue1, PropValue2);
    }
  }
}

TEST(PropertySetRegistryTest, PropertySetRegistry) {
  PropertySetRegistry PSR;
  PSR.add("Category1", "Prop1", 42);
  PSR.add("Category1", "Prop2", "Hello");
  SmallVector<int, 3> arr = {4, 16, 32};
  PSR.add("Category2", "A", arr);
  auto Serialized = PSR.writeJSON();
  auto PSR2 = PropertySetRegistry::readJSON({Serialized, ""});
  if (auto Err = PSR2.takeError())
    FAIL();
  checkEquality(PSR, *PSR2);
}
