#include "acxxel.h"
#include "config.h"
#include "gtest/gtest.h"

namespace {

using PlatformGetter = acxxel::Expected<acxxel::Platform *> (*)();
class MultiDeviceTest : public ::testing::TestWithParam<PlatformGetter> {};

TEST_P(MultiDeviceTest, AsyncCopy) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  int DeviceCount = Platform->getDeviceCount().getValue();
  EXPECT_GT(DeviceCount, 0);

  int Length = 3;
  auto A = std::unique_ptr<int[]>(new int[Length]);
  auto B0 = std::unique_ptr<int[]>(new int[Length]);
  auto B1 = std::unique_ptr<int[]>(new int[Length]);

  auto ASpan = acxxel::Span<int>(A.get(), Length);
  auto B0Span = acxxel::Span<int>(B0.get(), Length);
  auto B1Span = acxxel::Span<int>(B1.get(), Length);

  for (int I = 0; I < Length; ++I)
    A[I] = I;

  auto AsyncA = Platform->registerHostMem(ASpan).takeValue();
  auto AsyncB0 = Platform->registerHostMem(B0Span).takeValue();
  auto AsyncB1 = Platform->registerHostMem(B1Span).takeValue();

  acxxel::Stream Stream0 = Platform->createStream(0).takeValue();
  acxxel::Stream Stream1 = Platform->createStream(1).takeValue();
  auto Device0 = Platform->mallocD<int>(Length, 0).takeValue();
  auto Device1 = Platform->mallocD<int>(Length, 1).takeValue();

  EXPECT_FALSE(Stream0.asyncCopyHToD(AsyncA, Device0, Length)
                   .asyncCopyDToH(Device0, AsyncB0, Length)
                   .sync()
                   .isError());

  EXPECT_FALSE(Stream1.asyncCopyHToD(AsyncA, Device1, Length)
                   .asyncCopyDToH(Device1, AsyncB1, Length)
                   .sync()
                   .isError());

  for (int I = 0; I < Length; ++I) {
    EXPECT_EQ(B0[I], I);
    EXPECT_EQ(B1[I], I);
  }
}

TEST_P(MultiDeviceTest, Events) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  int DeviceCount = Platform->getDeviceCount().getValue();
  EXPECT_GT(DeviceCount, 0);

  acxxel::Stream Stream0 = Platform->createStream(0).takeValue();
  acxxel::Stream Stream1 = Platform->createStream(1).takeValue();
  acxxel::Event Event0 = Platform->createEvent(0).takeValue();
  acxxel::Event Event1 = Platform->createEvent(1).takeValue();

  EXPECT_FALSE(Stream0.enqueueEvent(Event0).sync().isError());
  EXPECT_FALSE(Stream1.enqueueEvent(Event1).sync().isError());

  EXPECT_TRUE(Event0.isDone());
  EXPECT_TRUE(Event1.isDone());

  EXPECT_FALSE(Event0.sync().isError());
  EXPECT_FALSE(Event1.sync().isError());
}

#if defined(ACXXEL_ENABLE_CUDA) || defined(ACXXEL_ENABLE_OPENCL)
INSTANTIATE_TEST_CASE_P(BothPlatformTest, MultiDeviceTest,
                        ::testing::Values(
#ifdef ACXXEL_ENABLE_CUDA
                            acxxel::getCUDAPlatform
#ifdef ACXXEL_ENABLE_OPENCL
                            ,
#endif
#endif
#ifdef ACXXEL_ENABLE_OPENCL
                            acxxel::getOpenCLPlatform
#endif
                            ));
#endif

} // namespace
