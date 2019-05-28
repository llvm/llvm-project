//==------------ image_impl.hpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/aligned_allocator.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/sycl_mem_obj.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {

enum class image_channel_order : unsigned int;
enum class image_channel_type : unsigned int;

namespace detail {

// utility functions and typedefs for image_impl
using image_allocator = aligned_allocator<byte, /*alignment*/ 64>;

// utility function: Returns the Number of Channels for a given Order.
uint8_t getImageNumberChannels(image_channel_order Order);

// utility function: Returns the number of bytes per image element
uint8_t getImageElementSize(uint8_t NumChannels, image_channel_type Type);

// validImageDataT: cl_int4, cl_uint4, cl_float4, cl_half4
// To be used in get_access method. Uncomment after get_access is implemented.
// template <typename T>
// using is_validImageDataT = typename detail::is_contained<
//    T, type_list<cl_int4, cl_uint4, cl_float4, cl_half4>>::type;

template <int Dimensions, typename AllocatorT = image_allocator>
class image_impl : public SYCLMemObjT {
private:
  template <bool B>
  using EnableIfPitchT =
      typename std::enable_if<B, range<Dimensions - 1>>::type;
  static_assert(Dimensions >= 1 || Dimensions <= 3,
                "Dimensions of cl::sycl::image can be 1, 2 or 3");

  void setPitches() {
    size_t WHD[3] = {1, 1, 1}; // Width, Height, Depth.
    for (int I = 0; I < Dimensions; I++)
      WHD[I] = MRange[I];
    MRowPitch = MElementSize * WHD[0];
    MSlicePitch = MRowPitch * WHD[1];
    MSizeInBytes = MSlicePitch * WHD[2];
  }

  template <bool B = (Dimensions > 1)>
  void setPitches(const EnableIfPitchT<B> Pitch) {
    MRowPitch = Pitch[0];
    MSlicePitch =
        (Dimensions == 3) ? Pitch[1] : MRowPitch; // Dimensions will be 2/3.
    // NumSlices is depth when dim==3, and height when dim==2.
    size_t NumSlices =
        (Dimensions == 3) ? MRange[2] : MRange[1]; // Dimensions will be 2/3.
    MSizeInBytes = MSlicePitch * NumSlices;
  }

  void handleHostData(void *HData) {
    MUserPtr = HData;
    // TO DO:
    //      Populate the function MUploadDataFn.
    //      Add the set_final_data function.
    //      Initialise allocated memory to the data HData points to (if needed).
    //      Some properties to be checked from MProps.
  }

  void handleHostData(const void *HData) {
    MHostPtrReadOnly = true;
    MUserPtr = const_cast<void *>(HData);
    // TO DO:
    //      Populate the function MUploadDataFn.
    //      Initialise allocated memory to the data HData points to (if needed).
    //      Some properties to be checked from MProps.
  }

  void handleHostData(shared_ptr_class<void> HData) {
    // MUserPtr = HData;
    // TO DO:
    //      Populate the function MUploadDataFn.
    //      Add the set_final_data function.
    //      Initialise allocated memory to the data HData points to (if needed).
    //      Some properties to be checked from MProps.
    //      Check if we need this specialized function at all.
  }

public:
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, PropList) {}

  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange, AllocatorT Allocator,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Allocator,
                   PropList) {}

  template <bool B = (Dimensions > 1)>
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Pitch, PropList) {}

  template <bool B = (Dimensions > 1)>
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, AllocatorT Allocator,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Pitch, Allocator,
                   PropList) {}

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    handleHostData(HData);
  }

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange, AllocatorT Allocator,
             const property_list &PropList = {})
      : MAllocator(Allocator), MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    handleHostData(HData);
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    handleHostData(HData);
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             AllocatorT Allocator, const property_list &PropList = {})
      : MAllocator(Allocator), MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    handleHostData(HData);
  }

  template <bool B = (Dimensions > 1)>
  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, const property_list &PropList = {})
      : MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    handleHostData(HData);
  }

  template <bool B = (Dimensions > 1)>
  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, AllocatorT Allocator,
             const property_list &PropList = {})
      : MAllocator(Allocator), MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    handleHostData(HData);
  }

  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    handleHostData(HData);
  }

  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             AllocatorT Allocator, const property_list &PropList = {})
      : MAllocator(Allocator), MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    handleHostData(HData);
  }

  /* Available only when: Dimensions > 1 */
  template <bool B = (Dimensions > 1)>
  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, const property_list &PropList = {})
      : MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    handleHostData(HData);
  }

  /* Available only when: Dimensions > 1 */
  template <bool B = (Dimensions > 1)>
  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, AllocatorT Allocator,
             const property_list &PropList = {})
      : MAllocator(Allocator), MProps(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    handleHostData(HData);
  }

  // Return a range object representing the size of the image in terms of the
  // number of elements in each dimension as passed to the constructor
  range<Dimensions> get_range() const { return MRange; }

  // Return a range object representing the pitch of the image in bytes.
  // Available only when: Dimensions == 2.
  template <bool B = (Dimensions == 2)>
  typename std::enable_if<B, range<1>>::type get_pitch() const {
    range<1> Temp = range<1>(MRowPitch);
    return Temp;
  }

  // Return a range object representing the pitch of the image in bytes.
  // Available only when: Dimensions == 3.
  template <bool B = (Dimensions == 3)>
  typename std::enable_if<B, range<2>>::type get_pitch() const {
    range<2> Temp = range<2>(MRowPitch, MSlicePitch);
    return Temp;
  }

  // Returns the size of the image storage in bytes
  size_t get_size() const { return MSizeInBytes; }

  // Returns the total number of elements in the image
  size_t get_count() const { return MRange.size(); }

  // Returns the allocator provided to the image
  AllocatorT get_allocator() const { return MAllocator; }

  template <typename propertyT> bool has_property() const {
    return MProps.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return MProps.get_property<propertyT>();
  }

  // TODO: Implement this function.
  void *allocateHostMem() override {
    if (true)
      throw cl::sycl::feature_not_supported(
          "HostMemoryAllocation Function Not Implemented for image class");
    return nullptr;
    // Implementation of the pure virtual function.
  }

  // TODO: Implement this function.
  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    cnri_event &OutEventToWait) override {
    if (true)
      throw cl::sycl::feature_not_supported(
          "MemoryAllocation Function Not Implemented for image class");
    return nullptr;
    // Implementation of the pure virtual function.
  }

  MemObjType getType() const override { return MemObjType::IMAGE; }

  // TODO: Implement this function.
  void releaseHostMem(void *Ptr) override {
    if (true)
      throw cl::sycl::feature_not_supported(
          "HostMemoryRelease Function Not Implemented for Image class");
    return;
    // Implementation of the pure virtual function.
  }

  // TODO: Implement this function.
  void releaseMem(ContextImplPtr Context, void *MemAllocation) override {
    if (true)
      throw cl::sycl::feature_not_supported(
          "MemoryRelease Function Not Implemented for Image class");
    return;
    // Implementation of the pure virtual function.
  }

private:
  bool MHostPtrReadOnly = false;
  AllocatorT MAllocator;
  std::function<void(void)> MUploadDataFn = nullptr;
  void *MUserPtr = nullptr;

  property_list MProps;
  range<Dimensions> MRange;
  image_channel_order MOrder;
  image_channel_type MType;
  uint8_t MNumChannels = 0; // Maximum Value - 4
  uint8_t MElementSize = 0; // Maximum Value - 16
  size_t MRowPitch = 0;
  size_t MSlicePitch = 0;
  size_t MSizeInBytes = 0;

}; // class image_impl

} // namespace detail

} // namespace sycl
} // namespace cl
