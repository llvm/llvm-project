// RUN: %clang -std=c++11 %s -o %t1.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %clang -std=c++11 -fsycl %s -o %t2.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out
//==-------image_constructors.cpp - SYCL image constructors basic test------==//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>

void no_delete(void *) {}

template <int Dims>
void test_constructors(cl::sycl::range<Dims> r, void *imageHostPtr) {

  cl::sycl::image_channel_order channelOrder =
      cl::sycl::image_channel_order::rgbx;
  cl::sycl::image_channel_type channelType =
      cl::sycl::image_channel_type::signed_int32;
  unsigned int elementSize = 12; // rgbx * i32
  int numElems = r.size();
  cl::sycl::property_list propList{}; // empty property list

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const property_list& = {})
   */
  {
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(imageHostPtr, channelOrder, channelType, r);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&, const property_list&)
   */
  {
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&, allocator,
   *              const property_list& = {})
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&, allocator,
   *              const property_list&)
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }
  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const property_list& = {})
   */
  {
    const auto constHostPtr = imageHostPtr;
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(constHostPtr, channelOrder, channelType, r);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<3>&, const property_list&)
   */
  {
    const auto constHostPtr = imageHostPtr;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        constHostPtr, channelOrder, channelType, r, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<3>&, allocator,
   *              const property_list& = {})
   */
  {
    const auto constHostPtr = imageHostPtr;
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        constHostPtr, channelOrder, channelType, r, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<3>&, allocator,
   *              const property_list&)
   */
  {
    const auto constHostPtr = imageHostPtr;
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        constHostPtr, channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const property_list& = {})
   */
  {
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(hostPointer, channelOrder, channelType, r);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&, const property_list&)
   */
  {
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(hostPointer, channelOrder,
                                                      channelType, r, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&, allocator,
   *              const property_list& = {})
   */
  {
    cl::sycl::image_allocator imgAlloc;
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(hostPointer, channelOrder,
                                                      channelType, r, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&, allocator,
   *              const property_list&)
   */
  {
    cl::sycl::image_allocator imgAlloc;
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        hostPointer, channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, const property_list& = {})
   */
  {
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, const property_list&)
   */
  {
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, allocator, const property_list& = {})
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, allocator, const property_list&)
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }
}

template <int Dims>
void test_constructors_with_pitch(cl::sycl::range<Dims> r, cl::sycl::range<Dims-1> pitch, void *imageHostPtr) {

  cl::sycl::image_channel_order channelOrder =
      cl::sycl::image_channel_order::rgbx;
  cl::sycl::image_channel_type channelType =
      cl::sycl::image_channel_type::signed_int32;
  unsigned int elementSize = 12; // rgbx * i32
  int numElems = r.size();
  cl::sycl::property_list propList{}; // empty property list


  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, const property_list& = {})
   */
  {
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, pitch);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, const property_list&)
   */
  {
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, pitch, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, allocator,
   *              const property_list& = {})
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, pitch, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, allocator, const property_list&)
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, pitch, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, const property_list& = {})
   */
  {
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(hostPointer, channelOrder, channelType, r, pitch);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, const property_list&)
   */
  {
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        hostPointer, channelOrder, channelType, r, pitch, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, allocator,
   *              const property_list& = {})
   */
  {
    cl::sycl::image_allocator imgAlloc;
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        hostPointer, channelOrder, channelType, r, pitch, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<3>&,
   *              const range<3 - 1>&, allocator, const property_list&)
   */
  {
    cl::sycl::image_allocator imgAlloc;
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        hostPointer, channelOrder, channelType, r, pitch, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, const range<3 - 1>&,
   *              const property_list& = {})
   */
  {
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r, pitch);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, const range<3 - 1>&,
   *              const property_list&)
   */
  {
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r, pitch, propList);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, const range<3 - 1>&, allocator,
   *              const property_list& = {})
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img =
        cl::sycl::image<Dims>(channelOrder, channelType, r, pitch, imgAlloc);
    assert(img.get_size() == (numElems * elementSize));
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<3>&, const range<3 - 1>&, allocator,
   *              const property_list&)
   */
  {
    cl::sycl::image_allocator imgAlloc;
    cl::sycl::image<Dims> img = cl::sycl::image<Dims>(
        channelOrder, channelType, r, pitch, imgAlloc, propList);
    assert(img.get_size() == (numElems * elementSize));
  }
}

int main() {

  int imageHostPtr[144]; // 16*9
  for (int i = 0; i < 144; i++)
    imageHostPtr[i] = i; // Maximum number of elements.

  // Ranges 
  cl::sycl::range<1> r1(3);
  cl::sycl::range<2> r2(3, 2);
  cl::sycl::range<3> r3(3, 2, 4);
  
  // Pitches
  cl::sycl::range<1> pitch2(36); // range is 3; elementSize = 12.
  cl::sycl::range<2> pitch3(36, 72); // range is 3,2; elementSize = 12.
  
  // Constructors without Pitch
  test_constructors<1>(r1, imageHostPtr);
  test_constructors<2>(r2, imageHostPtr);
  test_constructors<3>(r3, imageHostPtr);

  // Constructors with Pitch
  test_constructors_with_pitch<2>(r2, pitch2, imageHostPtr);
  test_constructors_with_pitch<3>(r3, pitch3, imageHostPtr);

  return 0;
}
