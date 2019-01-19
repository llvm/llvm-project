//===--- acxxel.cpp - Implementation details for the Acxxel API -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "acxxel.h"
#include "config.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace acxxel {

namespace cuda {
Expected<Platform *> getPlatform();
} // namespace cuda

namespace opencl {
Expected<Platform *> getPlatform();
} // namespace opencl

void logWarning(const std::string &Message) {
  std::cerr << "WARNING: " << Message << "\n";
}

Expected<Platform *> getCUDAPlatform() {
#ifdef ACXXEL_ENABLE_CUDA
  return cuda::getPlatform();
#else
  return Status("library was build without CUDA support");
#endif
}

Expected<Platform *> getOpenCLPlatform() {
#ifdef ACXXEL_ENABLE_OPENCL
  return opencl::getPlatform();
#else
  return Status("library was build without OpenCL support");
#endif
}

Stream::Stream(Stream &&) noexcept = default;
Stream &Stream::operator=(Stream &&) noexcept = default;

Status Stream::sync() {
  return takeStatusOr(ThePlatform->streamSync(TheHandle.get()));
}

Status Stream::waitOnEvent(Event &Event) {
  return takeStatusOr(ThePlatform->streamWaitOnEvent(
      TheHandle.get(), ThePlatform->getEventHandle(Event)));
}

Stream &
Stream::addCallback(std::function<void(Stream &, const Status &)> Callback) {
  setStatus(ThePlatform->addStreamCallback(*this, std::move(Callback)));
  return *this;
}

Stream &Stream::asyncKernelLaunch(const Kernel &TheKernel,
                                  KernelLaunchDimensions LaunchDimensions,
                                  Span<void *> Arguments,
                                  Span<size_t> ArgumentSizes,
                                  size_t SharedMemoryBytes) {
  setStatus(ThePlatform->rawEnqueueKernelLaunch(
      TheHandle.get(), TheKernel.TheHandle.get(), LaunchDimensions, Arguments,
      ArgumentSizes, SharedMemoryBytes));
  return *this;
}

Stream &Stream::enqueueEvent(Event &E) {
  setStatus(ThePlatform->enqueueEvent(ThePlatform->getEventHandle(E),
                                      TheHandle.get()));
  return *this;
}

Event::Event(Event &&) noexcept = default;
Event &Event::operator=(Event &&) noexcept = default;

bool Event::isDone() { return ThePlatform->eventIsDone(TheHandle.get()); }

Status Event::sync() { return ThePlatform->eventSync(TheHandle.get()); }

Expected<float> Event::getSecondsSince(const Event &Previous) {
  Expected<float> MaybeSeconds = ThePlatform->getSecondsBetweenEvents(
      Previous.TheHandle.get(), TheHandle.get());
  if (MaybeSeconds.isError())
    MaybeSeconds.getError();
  return MaybeSeconds;
}

Expected<Kernel> Program::createKernel(const std::string &Name) {
  Expected<void *> MaybeKernelHandle =
      ThePlatform->rawCreateKernel(TheHandle.get(), Name);
  if (MaybeKernelHandle.isError())
    return MaybeKernelHandle.getError();
  return Kernel(ThePlatform, MaybeKernelHandle.getValue(),
                ThePlatform->getKernelHandleDestructor());
}

Program::Program(Program &&) noexcept = default;
Program &Program::operator=(Program &&That) noexcept = default;

Kernel::Kernel(Kernel &&) noexcept = default;
Kernel &Kernel::operator=(Kernel &&That) noexcept = default;

} // namespace acxxel
