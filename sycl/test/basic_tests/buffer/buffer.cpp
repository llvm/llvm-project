// RUN: %clang -std=c++11 %s -o %t1.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %clang -std=c++11 -fsycl %s -o %t2.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out
//==------------------- buffer.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

int main() {
  int data = 5;
  bool failed = false;
  buffer<int, 1> buf(&data, range<1>(1));
  {
    int data1[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    {
      buffer<int, 1> b(data1, range<1>(10), {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class init_a>(range<1>{10},
                                       [=](id<1> index) { B[index] = 0; });
      });

    } // Data is copied back because there is a user side shared_ptr
    for (int i = 0; i < 10; i++)
      assert(data1[i] == 0);
  }

  {
    std::vector<int> data1(10, -1);
    {
      buffer<int, 1> b(data1.data(), range<1>(10));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class init_b>(range<1>{10},
                                       [=](id<1> index) { B[index] = 0; });
      });

    } // Data is copied back because there is a user side shared_ptr
    for (int i = 0; i < 10; i++)
      assert(data1[i] == 0);
  }

  {
    const size_t bufsSize = 10;
    vector_class<int> res(bufsSize);
    shared_ptr_class<int> ptr1{new int[bufsSize], [](int *p) { delete[] p; }};
    for (int *ptr = ptr1.get(), *end = ptr + bufsSize; ptr < end; ++ptr) {
      *ptr = -1;
    }
    {
      buffer<int, 1> b(ptr1, range<1>(bufsSize));
      buffer<int, 1> c(ptr1, range<1>(bufsSize));
      buffer<int, 1> d((range<1>(bufsSize)));
      buffer<int, 1> e(res.data(), range<1>(bufsSize));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        auto C = c.get_access<access::mode::read_write>(cgh);
        auto D = d.get_access<access::mode::write>(cgh);
        auto E = e.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class init_c>(range<1>{bufsSize}, [=](id<1> index) {
          B[index]++;
          C[index]++;
          D[index] = C[index] + B[index] + 1;
          E[index] = D[index] * (B[index] + 1) - 1;
        });
      });
    } // Data is copied back because there is a user side shared_ptr
    for (int i = 0; i < bufsSize; i++) {
      assert(ptr1.get()[i] == 0);
      assert(res[i] == 0);
    }
  }

  {
    std::cout << "move constructor" << std::endl;
    int data = 5;
    buffer<int, 1> Buffer(&data, range<1>(1));
    size_t hash = std::hash<buffer<int, 1>>()(Buffer);
    buffer<int, 1> MovedBuffer(std::move(Buffer));
    assert(hash == (std::hash<buffer<int, 1>>()(MovedBuffer)));
    assert(MovedBuffer.get_range() == range<1>(1));
    assert(MovedBuffer.get_size() == (sizeof(data) * 1));
    assert(MovedBuffer.get_count() == 1);
  }

  {
    std::cout << "move assignment operator" << std::endl;
    int data = 5;
    buffer<int, 1> Buffer(&data, range<1>(1));
    size_t hash = std::hash<buffer<int, 1>>()(Buffer);
    int data_2 = 4;
    buffer<int, 1> WillMovedBuffer(&data, range<1>(1));
    WillMovedBuffer = std::move(Buffer);
    assert(hash == (std::hash<buffer<int, 1>>()(WillMovedBuffer)));
    assert(WillMovedBuffer.get_range() == range<1>(1));
    assert(WillMovedBuffer.get_size() == (sizeof(data) * 1));
    assert(WillMovedBuffer.get_count() == 1);
  }

  {
    std::cout << "copy constructor" << std::endl;
    int data = 5;
    buffer<int, 1> Buffer(&data, range<1>(1));
    size_t hash = std::hash<buffer<int, 1>>()(Buffer);
    buffer<int, 1> BufferCopy(Buffer);
    assert(hash == (std::hash<buffer<int, 1>>()(Buffer)));
    assert(hash == (std::hash<buffer<int, 1>>()(BufferCopy)));
    assert(Buffer == BufferCopy);
    assert(BufferCopy.get_range() == range<1>(1));
    assert(BufferCopy.get_size() == (sizeof(data) * 1));
    assert(BufferCopy.get_count() == 1);
  }

  {
    std::cout << "copy assignment operator" << std::endl;
    int data = 5;
    buffer<int, 1> Buffer(&data, range<1>(1));
    size_t hash = std::hash<buffer<int, 1>>()(Buffer);
    int data_2 = 4;
    buffer<int, 1> WillBufferCopy(&data_2, range<1>(1));
    WillBufferCopy = Buffer;
    assert(hash == (std::hash<buffer<int, 1>>()(Buffer)));
    assert(hash == (std::hash<buffer<int, 1>>()(WillBufferCopy)));
    assert(Buffer == WillBufferCopy);
    assert(WillBufferCopy.get_range() == range<1>(1));
    assert(WillBufferCopy.get_size() == (sizeof(data) * 1));
    assert(WillBufferCopy.get_count() == 1);
  }

  auto checkAllOf = [](const int *const array, size_t n, int v, int line) {
    for (size_t i = 0; i < n; ++i) {
      if (array[i] != v) {
        std::cout << "line: " << line << " array[" << i << "] is " << array[i]
                  << " expected " << v << std::endl;
        assert(false);
      }
    }
  };

  {
    int data[10] = {0};
    int result[10] = {0};
    {
      buffer<int, 1> Buffer(data, range<1>(10));
      Buffer.set_final_data(nullptr);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class Nullptr>(range<1>{10},
                                        [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result, 10, 0, __LINE__);
  }
  {
    int data[10] = {0};
    int result[10] = {0};
    {
      buffer<int, 1> Buffer(data, range<1>(10));
      Buffer.set_final_data(result);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class rawPointer>(range<1>{10},
                                           [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result, 10, 1, __LINE__);
  }
  {
    int data[10] = {0};
    std::shared_ptr<int> result(new int[10]());
    {
      buffer<int, 1> Buffer(data, range<1>(10));
      std::weak_ptr<int> resultWeak = result;
      Buffer.set_final_data(resultWeak);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class sharedPointer>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result.get(), 10, 1, __LINE__);
  }

  {
    int data[10] = {0};
    int result[10] = {0};
    // Creation of shared_ptr from a static array
    // It's need for for check that a copyback did not happen
    std::shared_ptr<int> resultShared(result, [](int const *) {});
    {
      buffer<int, 1> Buffer(data, range<1>(10));
      std::weak_ptr<int> resultWeak = resultShared;
      Buffer.set_final_data(resultWeak);
      queue myQueue;
      resultShared.reset();
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class sharedPointerAndReset>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    assert(resultShared.get() == nullptr);
    checkAllOf(result, 10, 0, __LINE__);
  }
  {
    int data[10] = {0};
    std::vector<int> result(10, 0);
    {
      buffer<int, 1> Buffer(data, range<1>(10));
      Buffer.set_final_data(result.begin());
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class vectorIterator>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result.data(), 10, 1, __LINE__);
  }
  {
    int data[10] = {0};
    int result[10] = {0};
    {
      buffer<int, 1> Buffer(data, range<1>(10),
                            {property::buffer::use_host_ptr()});
      Buffer.set_final_data(nullptr);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class nullptAndUseHost>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result, 10, 0, __LINE__);
  }
  {
    int data[10] = {0};
    int result[10] = {0};
    {
      buffer<int, 1> Buffer(data, range<1>(10),
                            {property::buffer::use_host_ptr()});
      Buffer.set_final_data(result);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class rawPointerAndUseHost>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result, 10, 1, __LINE__);
  }
  {
    int data[10] = {0};
    std::shared_ptr<int> result(new int[10]());
    {
      buffer<int, 1> Buffer(data, range<1>(10),
                            {property::buffer::use_host_ptr()});
      std::weak_ptr<int> resultWeak = result;
      Buffer.set_final_data(resultWeak);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class sharedPointerUseHost>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result.get(), 10, 1, __LINE__);
  }
  {
    int data[10] = {0};
    int result[10] = {0};
    // Creation of shared_ptr from a static array
    // It's need for for check that a copyback did not happen
    std::shared_ptr<int> resultShared(result, [](int const *) {});
    {
      buffer<int, 1> Buffer(data, range<1>(10),
                            {property::buffer::use_host_ptr()});
      std::weak_ptr<int> resultWeak = resultShared;
      Buffer.set_final_data(resultWeak);
      queue myQueue;
      resultShared.reset();
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class sharedPointerAndResetUseHost>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    assert(resultShared.get() == nullptr);
    checkAllOf(result, 10, 0, __LINE__);
  }
  {
    int data[10] = {0};
    std::vector<int> result(10, 0);
    {
      buffer<int, 1> Buffer(data, range<1>(10),
                            {property::buffer::use_host_ptr()});
      Buffer.set_final_data(result.begin());
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class vectorIteratorAndUseHost>(
            range<1>{10}, [=](id<1> index) { B[index] = 1; });
      });
    }
    checkAllOf(result.data(), 10, 1, __LINE__);
  }

  {
    int result[20][20] = {0};
    {
      buffer<int, 2> Buffer(range<2>(20, 20));
      Buffer.set_final_data((int *)result);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class bufferByRange2Init>(
            range<2>{20, 20}, [=](id<2> index) { B[index] = 0; });
      });

      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class bufferByRange2>(
            range<2>{10, 10}, [=](id<2> index) { B[index] = 1; });
      });
    }

    for (size_t i = 0; i < 20; ++i) {
      for (size_t j = 0; j < 20; ++j) {
        if (i < 10 && j < 10) {
          if (result[i][j] != 1) {
            std::cout << "line: " << __LINE__ << " result[" << i << "][" << j
                      << "] is " << result[i][j] << " expected " << 1
                      << std::endl;
            assert(false);
          }
        } else {
          if (result[i][j] != 0) {
            std::cout << "line: " << __LINE__ << " result[" << i << "][" << j
                      << "] is " << result[i][j] << " expected " << 0
                      << std::endl;
            assert(false);
          }
        }
      }
    }
  }

  {
    int result[20][20] = {0};
    {
      buffer<int, 2> Buffer(range<2>(20, 20));
      Buffer.set_final_data((int *)result);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = Buffer.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class bufferByRangeOffsetInit>(
            range<2>{20, 20}, [=](id<2> index) { B[index] = 0; });
      });

      myQueue.submit([&](handler &cgh) {
        accessor<int, 2, access::mode::write, access::target::global_buffer,
               access::placeholder::false_t>
          B(Buffer, cgh, range<2>(20,20), id<2>(10,10));
        cgh.parallel_for<class bufferByRangeOffset>(
            range<2>{10, 5}, [=](id<2> index) { B[index] = 1; });
      });
    }

    for (size_t i = 0; i < 20; ++i) {
      for (size_t j = 0; j < 20; ++j) {
        if (i >= 10 && j >= 10 && j < 15) {
          if (result[i][j] != 1) {
            std::cout << "line: " << __LINE__ << " result[" << i << "][" << j
                      << "] is " << result[i][j] << " expected " << 1
                      << std::endl;
            assert(false);
          }
        } else {
          if (result[i][j] != 0) {
            std::cout << "line: " << __LINE__ << " result[" << i << "][" << j
                      << "] is " << result[i][j] << " expected " << 0
                      << std::endl;
            assert(false);
          }
        }
      }
    }
  }

  {
    std::vector<int> data1(10, -1);
    {
      buffer<int, 1> b(data1.begin() + 2, data1.begin() + 5);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class iter_constuctor>(
            range<1>{3}, [=](id<1> index) { B[index] = 20; });
      });
    }
    // Data is copied back in the desctruction of the buffer created from
    // pair of non-const iterators
    for (int i = 0; i < 2; i++)
      assert(data1[i] == -1);
    for (int i = 2; i < 5; i++)
      assert(data1[i] == 20);
    for (int i = 5; i < 10; i++)
      assert(data1[i] == -1);
  }

  // Check that data is not copied back in the desctruction of the buffer
  // created from pair of const iterators
  {
    std::vector<int> data1(10, -1);
    {
      buffer<int, 1> b(data1.cbegin() + 2, data1.cbegin() + 5);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class const_iter_constuctor>(
            range<1>{3}, [=](id<1> index) { B[index] = 20; });
      });
    }
    for (int i = 0; i < 10; i++)
      assert(data1[i] == -1);
  }

  // Try use_host_pointer for the buffer created from
  {
    std::vector<int> data1(10, -1);
    {
      buffer<int, 1> b(data1.begin() + 2, data1.begin() + 5,
                       {property::buffer::use_host_ptr()});
      b.set_final_data(data1.begin() + 2);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class iter_constuctor_use_host_ptr>(
            range<1>{3}, [=](id<1> index) { B[index] = 20; });
      });
    }
    for (int i = 0; i < 2; i++)
      assert(data1[i] == -1);
    for (int i = 2; i < 5; i++)
      assert(data1[i] == 20);
    for (int i = 5; i < 10; i++)
      assert(data1[i] == -1);
  }

  // Check that data is copied back when using set_final_data for the buffer
  // created from pair of iterators
  {
    std::vector<int> data1(10, -1);
    std::vector<int> data2(10, -1);
    {
      buffer<int, 1> b(data1.begin() + 2, data1.begin() + 5);
      b.set_final_data(data2.begin() + 2);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class iter_constuctor_set_final_data>(
            range<1>{3}, [=](id<1> index) { B[index] = 20; });
      });
    }
    for (int i = 0; i < 2; i++)
      assert(data2[i] == -1);
    for (int i = 2; i < 5; i++)
      assert(data2[i] == 20);
    for (int i = 5; i < 10; i++)
      assert(data2[i] == -1);
  }

  // Check that data is copied back after forcing write-back using
  // set_write_back
  {
    std::vector<int> data1(10, -1);
    {
      buffer<int, 1> b(range<1>(10));
      b.set_final_data(data1.data());
      b.set_write_back(true);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class wb>(range<1>{10},
                                       [=](id<1> index) { B[index] = 0; });
      });

    }
    // Data is copied back because there is a user side ptr and write-back is
    // enabled
    for (int i = 0; i < 10; i++)
      if (data1[i] != 0) {
        assert(false);
        failed = true;
      }
  }

  // Check that data is not copied back after canceling write-back using
  // set_write_back
  {
    std::vector<int> data1(10, -1);
    {
      buffer<int, 1> b(range<1>(10));
      b.set_final_data(data1.data());
      b.set_write_back(false);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class notwb>(range<1>{10},
                                       [=](id<1> index) { B[index] = 0; });
      });

    }
    // Data is not copied back because write-back is canceled
    for (int i = 0; i < 10; i++)
      if (data1[i] != -1) {
        assert(false);
        failed = true;
      }
  }

  {
    std::vector<int> data1(10, -1);
    std::vector<int> data2(10, -2);
    {
      buffer<int, 1> a(data1.data(), range<1>(10));
      buffer<int, 1> b(data2.data(), range<1>(10));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto A = a.get_access<access::mode::read_write>(cgh);
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class override_lambda>(
            range<1>{10}, [=](id<1> index) { A[index] = 0; });
      });
    } // Data is copied back
    for (int i = 0; i < 10; i++)
      assert(data2[i] == -2);
    for (int i = 0; i < 10; i++)
      assert(data1[i] == 0);
  }

  {
    queue myQueue;
    if (!myQueue.is_host()) {
      std::vector<int> data1(10, -1);
      std::vector<int> data2(10, -2);
      {
        buffer<int, 1> a(data1.data(), range<1>(10));
        buffer<int, 1> b(data2.data(), range<1>(10));

        program prog(myQueue.get_context());
        prog.build_with_source("kernel void override_source(global int* Acc) "
                               "{Acc[get_global_id(0)] = 0; }\n");
        cl::sycl::kernel krn = prog.get_kernel("override_source");
        myQueue.submit([&](handler &cgh) {
          auto A = a.get_access<access::mode::read_write>(cgh);
          cgh.set_arg(0, A);
          auto B = b.get_access<access::mode::read_write>(cgh);
          cgh.parallel_for(cl::sycl::range<1>(10), krn);
        });
      } // Data is copied back
      for (int i = 0; i < 10; i++)
        assert(data2[i] == -2);
      for (int i = 0; i < 10; i++)
        assert(data1[i] == 0);
    }
  }

  {
    std::vector<int> data1(10, -1);
    std::vector<int> data2(10, -2);
    {
      buffer<int, 1> a(data1.data(), range<1>(10));
      buffer<int, 1> b(data2.data(), range<1>(10));
      accessor<int, 1, access::mode::read_write, access::target::global_buffer,
               access::placeholder::true_t>
          A(a);
      accessor<int, 1, access::mode::read_write, access::target::global_buffer,
               access::placeholder::true_t>
          B(b);
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        cgh.require(A);
        cgh.require(B);
        cgh.parallel_for<class override_lambda_placeholder>(
            range<1>{10}, [=](id<1> index) { A[index] = 0; });
      });
    } // Data is copied back
    for (int i = 0; i < 10; i++)
      assert(data2[i] == -2);
    for (int i = 0; i < 10; i++)
      assert(data1[i] == 0);
  }

  {
    queue myQueue;
    if (!myQueue.is_host()) {
      std::vector<int> data1(10, -1);
      std::vector<int> data2(10, -2);
      {
        buffer<int, 1> a(data1.data(), range<1>(10));
        buffer<int, 1> b(data2.data(), range<1>(10));
        accessor<int, 1, access::mode::read_write,
                 access::target::global_buffer, access::placeholder::true_t>
            A(a);
        accessor<int, 1, access::mode::read_write,
                 access::target::global_buffer, access::placeholder::true_t>
            B(b);

        program prog(myQueue.get_context());
        prog.build_with_source("kernel void override_source_placeholder(global "
                               "int* Acc) {Acc[get_global_id(0)] = 0; }\n");
        cl::sycl::kernel krn = prog.get_kernel("override_source_placeholder");

        myQueue.submit([&](handler &cgh) {
          cgh.require(A);
          cgh.set_arg(0, A);
          cgh.require(B);
          cgh.parallel_for(cl::sycl::range<1>(10), krn);
        });
      } // Data is copied back
      for (int i = 0; i < 10; i++)
        assert(data2[i] == -2);
      for (int i = 0; i < 10; i++)
        assert(data1[i] == 0);
    }
  }

  // TODO tests with mutex property
  return failed;
}
