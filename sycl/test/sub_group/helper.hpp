//==---------- helper.hpp - SYCL sub_group helper functions ----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
template <typename T> void exit_if_not_equal(T val, T ref, const char *name) {
  if (std::fabs(val - ref) > 0.01) {
    std::cout << "Unexpected result for " << name << ": " << val
              << " expected value: " << ref << std::endl;
    exit(1);
  }
}
/* CPU returns max number of SG, GPU returns mux SG size for
 * CL_DEVICE_MAX_NUM_SUB_GROUPS device parameter. This function aligns the
 * value.
 * */
inline size_t get_sg_size(const device &Device) {
  size_t max_num_sg = Device.get_info<info::device::max_num_sub_groups>();
  if (Device.get_info<info::device::device_type>() == info::device_type::cpu) {
    size_t max_wg_size = Device.get_info<info::device::max_work_group_size>();
    return max_wg_size / max_num_sg;
  }
  if (Device.get_info<info::device::device_type>() == info::device_type::gpu) {
    return max_num_sg;
  }
  std::cout << "Unexpected deive type" << std::endl;
  exit(1);
}

bool core_sg_supported(const device &Device) {
  return (Device.has_extension("cl_khr_subgroups") ||
          Device.get_info<info::device::version>().find(" 2.1") !=
              string_class::npos);
}
