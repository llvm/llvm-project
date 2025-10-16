#include "InternalEvent.h"
#include <omp-tools.h>
#include <sstream>

#include "gtest/gtest.h"

using namespace omptest;

TEST(InternalEvent_equality_ops, Dispatch_identity) {
  ompt_data_t DI{.value = 31};
  internal::Dispatch D{/*ParallelData=*/(ompt_data_t *)0x11,
                       /*TaskData=*/(ompt_data_t *)0x22,
                       /*Kind=*/ompt_dispatch_t::ompt_dispatch_iteration,
                       /*Instance=*/DI};

  EXPECT_EQ(D == D, true);
}

TEST(InternalEvent_equality_ops, Dispatch_same) {
  ompt_data_t DI{.ptr = (void *)0x33};
  internal::Dispatch D1{/*ParallelData=*/(ompt_data_t *)0x11,
                        /*TaskData=*/(ompt_data_t *)0x22,
                        /*Kind=*/ompt_dispatch_t::ompt_dispatch_section,
                        /*Instance=*/DI};

  internal::Dispatch D2{/*ParallelData=*/(ompt_data_t *)0x11,
                        /*TaskData=*/(ompt_data_t *)0x22,
                        /*Kind=*/ompt_dispatch_t::ompt_dispatch_section,
                        /*Instance=*/DI};

  EXPECT_EQ(D1 == D2, true);
}

TEST(InternalEvent_equality_ops, Dispatch_different_kind) {
  ompt_data_t DI{.ptr = (void *)0x33};
  internal::Dispatch D1{/*ParallelData=*/(ompt_data_t *)0x11,
                        /*TaskData=*/(ompt_data_t *)0x22,
                        /*Kind=*/ompt_dispatch_t::ompt_dispatch_section,
                        /*Instance=*/DI};

  internal::Dispatch D2{/*ParallelData=*/(ompt_data_t *)0x11,
                        /*TaskData=*/(ompt_data_t *)0x22,
                        /*Kind=*/ompt_dispatch_t::ompt_dispatch_iteration,
                        /*Instance=*/DI};

  // Demonstrate that 'Kind' is the only relevant field for equality.
  EXPECT_EQ(D1 == D2, false);
}

TEST(InternalEvent_equality_ops, Dispatch_same_kind_different_other) {
  ompt_data_t DI1{.ptr = (void *)0x33};
  internal::Dispatch D1{/*ParallelData=*/(ompt_data_t *)0x11,
                        /*TaskData=*/(ompt_data_t *)0x22,
                        /*Kind=*/ompt_dispatch_t::ompt_dispatch_section,
                        /*Instance=*/DI1};

  ompt_data_t DI2{.ptr = (void *)0x66};
  internal::Dispatch D2{/*ParallelData=*/(ompt_data_t *)0x44,
                        /*TaskData=*/(ompt_data_t *)0x55,
                        /*Kind=*/ompt_dispatch_t::ompt_dispatch_section,
                        /*Instance=*/DI2};

  // Demonstrate that 'Kind' is the only relevant field for equality.
  EXPECT_EQ(D1 == D2, true);
}
