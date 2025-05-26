
//  Copyright John Maddock 2016.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CUDA_MANAGED_PTR_HPP
#define BOOST_MATH_CUDA_MANAGED_PTR_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <cuda_runtime.h>

class managed_holder_base
{
protected:
   static int count;
   managed_holder_base() { ++count; }
   ~managed_holder_base()
   {
      if(0 == --count)
         cudaDeviceSynchronize();
   }
};

int managed_holder_base::count = 0;

//
// Reset the device and exit:
// cudaDeviceReset causes the driver to clean up all state. While
// not mandatory in normal operation, it is good practice.  It is also
// needed to ensure correct operation when the application is being
// profiled. Calling cudaDeviceReset causes all profile data to be
// flushed before the application exits.
//
// We have a global instance of this class, plus instances for each
// managed pointer.  Last one out the door switches the lights off.
//
class cudaResetter
{
   static int count;
public:
   cudaResetter() { ++count;  }
   ~cudaResetter()
   {
      if(--count == 0)
      {
         cudaError_t err = cudaDeviceReset();
         if(err != cudaSuccess)
         {
            std::cerr << "Failed to deinitialize the device! error=" << cudaGetErrorString(err) << std::endl;
         }
      }
   }
};

int cudaResetter::count = 0;

cudaResetter global_resetter;

template <class T>
class cuda_managed_ptr
{
   T* data;
   static const cudaResetter resetter;
   cuda_managed_ptr(const cuda_managed_ptr&) = delete;
   cuda_managed_ptr& operator=(cuda_managed_ptr const&) = delete;
   void free()
   {
      if(data)
      {
         cudaDeviceSynchronize();
         cudaError_t err = cudaFree(data);
         if(err != cudaSuccess)
         {
            std::cerr << "Failed to deinitialize the device! error=" << cudaGetErrorString(err) << std::endl;
         }
      }
   }
public:
   cuda_managed_ptr() : data(0) {}
   cuda_managed_ptr(std::size_t n)
   {
      cudaError_t err = cudaSuccess;
      void *ptr;
      err = cudaMallocManaged(&ptr, n * sizeof(T));
      if(err != cudaSuccess)
         throw std::runtime_error(cudaGetErrorString(err));
      cudaDeviceSynchronize();
      data = static_cast<T*>(ptr);
   }
   cuda_managed_ptr(cuda_managed_ptr&& o)
   {
      data = o.data;
      o.data = 0;
   }
   cuda_managed_ptr& operator=(cuda_managed_ptr&& o)
   {
      free();
      data = o.data;
      o.data = 0;
      return *this;
   }
   ~cuda_managed_ptr()
   {
      free();
   }

   class managed_holder : managed_holder_base
   {
      T* pdata;
   public:
      managed_holder(T* p) : managed_holder_base(), pdata(p) {}
      managed_holder(const managed_holder& o) : managed_holder_base(), pdata(o.pdata) {}
      operator T* () { return pdata; }
      T& operator[] (std::size_t n) { return pdata[n]; }
   };
   class const_managed_holder : managed_holder_base
   {
      const T* pdata;
   public:
      const_managed_holder(T* p) : managed_holder_base(), pdata(p) {}
      const_managed_holder(const managed_holder& o) : managed_holder_base(), pdata(o.pdata) {}
      operator const T* () { return pdata; }
      const T& operator[] (std::size_t n) { return pdata[n]; }
   };

   managed_holder get() { return managed_holder(data); }
   const_managed_holder get()const { return data; }
   T& operator[](std::size_t n) { return data[n]; }
   const T& operator[](std::size_t n)const { return data[n]; }
};

template <class T>
cudaResetter const cuda_managed_ptr<T>::resetter;

#endif
