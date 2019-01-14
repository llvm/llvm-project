// RUN: %clang -std=c++11 -fsycl %s -o %t.run -lOpenCL -lsycl -lstdc++
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUNx (TODO: nd_item::barrier() is not implemented on HOST): env SYCL_DEVICE_TYPE=HOST %t.run

//==--------device_event.cpp - SYCL class device_event test ----------------==//
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

// Define the number of work items to enqueue.
const int nElems = 1024*1024u;
const int workGroupSize = 16;

// Check the result is correct.
int check_results(int *data, size_t stride) {
  int result = 0;
  int earlyFailout = 1000;
  for (int i = 0; i < nElems; i += workGroupSize) {
    int copiedVal = i;
    for (int j = 0; j < workGroupSize; j++) {
      int expectedVal;
      if ((j % stride) == 0) {
        expectedVal = 300 + copiedVal;
        copiedVal++;
      }
      else {
        expectedVal = i + j + 700;
      }
      if (data[i + j] != expectedVal) {
        std::cout << "fail: stride = " << stride << ", "
                  << "data[" << i + j << "] = " << data[i + j]
                  << "; expected result = " << expectedVal << "\n";
        result = 1;
        if (--earlyFailout == 0) {
          return result;
        }
      }
    }
  }
  return result;
}

int test_strideN(size_t stride) {
  // Define and initialize data to be copied to the device.
  int out_data[nElems] = {0};
  int nElemsToCopy = (workGroupSize / stride);
  if (workGroupSize % stride)
    nElemsToCopy++;

  try {
    default_selector selector;
    queue myQueue(selector, [](exception_list l) {
      for (auto ep : l) {
        try {
          std::rethrow_exception(ep);
        } catch (std::exception& e) {
          std::cout << e.what();
        }
      }
    });

    buffer<int, 1> out_buf(out_data, range<1>(nElems));

    myQueue.submit([&](handler& cgh) {

      auto out_ptr = out_buf.get_access<access::mode::write>(cgh);
      accessor<cl::sycl::cl_int, 1, access::mode::read_write, access::target::local>
          local_acc(range<1>(16), cgh);

      // Create work-groups with 16 work items in each group.
      auto myRange = nd_range<1>(range<1>(nElems), range<1>(workGroupSize));

      auto myKernel = ([=](nd_item<1> item) {
        // Write the values 300, 301, ...., 363 to local memory.
        // We expect to see these values in global memory
        // after async mem copy that is done below.
        local_acc[item.get_local_id()] = item.get_global_id()[0] + 300;

        auto grp = item.get_group();
        local_ptr<int> lptr = local_acc.get_pointer();
        global_ptr<int> gptr = out_ptr.get_pointer() + grp.get_id()[0] * 16;

        // Write the values 700, 701, ..., 763 to global memory.
        // Why? Well, a) to ensure that something is written into that memory
        // inside the work item; b) check possible mem write crazy effects,
        // that are not supposed to happen, but who knows..., c) to see those
        // values at the end if something goes wrong during the ASYNC MEM COPY.
        out_ptr[item.get_global_id()[0]] = item.get_global_id()[0] + 700;

        item.barrier();

        // Copy from local memory to global memory.
        device_event dev_event = grp.async_work_group_copy(gptr, lptr, nElemsToCopy, stride);
        grp.wait_for(dev_event);
      });

      cgh.parallel_for<class assign_elements>(myRange, myKernel);
    });

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 2;
  }

  return check_results(out_data, stride);
}

int main() {
  for (int i = 1; i < workGroupSize; i++) {
    int result = test_strideN(i);
    if (result)
      return result;
  }

  return 0;
}
