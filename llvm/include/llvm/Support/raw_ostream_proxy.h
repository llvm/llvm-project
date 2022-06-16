//===- raw_ostream_proxy.h - Proxies for raw output streams -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_OSTREAM_PROXY_H
#define LLVM_SUPPORT_RAW_OSTREAM_PROXY_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// Common bits for \a raw_ostream_proxy_adaptor<>, split out to dedup in
/// template instantions.
class raw_ostream_proxy_adaptor_base {
protected:
  raw_ostream_proxy_adaptor_base() = delete;
  raw_ostream_proxy_adaptor_base(const raw_ostream_proxy_adaptor_base &) =
      delete;

  explicit raw_ostream_proxy_adaptor_base(raw_ostream &OS)
      : OS(&OS), PreferredBufferSize(OS.GetBufferSize()) {
    // Drop OS's buffer to make this->flush() forward. This proxy will add a
    // buffer in its place.
    OS.SetUnbuffered();
  }

  ~raw_ostream_proxy_adaptor_base() {
    assert(!OS && "Derived objects should call resetProxiedOS()");
  }

  /// Stop proxying the stream, taking the derived object by reference as \p
  /// ThisProxyOS.  Updates \p ThisProxyOS to stop buffering before setting \a
  /// OS to \c nullptr, ensuring that future writes crash immediately.
  void resetProxiedOS(raw_ostream &ThisProxyOS) {
    ThisProxyOS.SetUnbuffered();
    OS = nullptr;
  }

  bool hasProxiedOS() const { return OS; }
  raw_ostream &getProxiedOS() const {
    assert(OS && "raw_ostream_proxy_adaptor use after reset");
    return *OS;
  }
  size_t getPreferredBufferSize() const { return PreferredBufferSize; }

private:
  raw_ostream *OS;

  /// Caches the value of OS->GetBufferSize() at construction time.
  size_t PreferredBufferSize;
};

/// Adaptor to create a stream class that proxies another \a raw_ostream.
///
/// Use \a raw_ostream_proxy_adaptor<> directly to implement an abstract
/// derived class of \a raw_ostream as a proxy. Otherwise use \a
/// raw_ostream_proxy.
///
/// Most operations are forwarded to the proxied stream.
///
/// If the proxied stream is buffered, the buffer is dropped and moved to this
/// stream. This allows \a flush() to work correctly, flushing immediately from
/// the proxy through to the final stream, and avoids any wasteful
/// double-buffering.
///
/// \a enable_colors() changes both the proxied stream and the proxy itself.
/// \a is_displayed() and \a has_colors() are forwarded to the proxy. \a
/// changeColor(), resetColor(), and \a reverseColor() are not forwarded, since
/// they need to call \a flush() and the buffer lives in the proxy.
template <class RawOstreamT = raw_ostream>
class raw_ostream_proxy_adaptor : public RawOstreamT,
                                  public raw_ostream_proxy_adaptor_base {
  void write_impl(const char *Ptr, size_t Size) override {
    getProxiedOS().write(Ptr, Size);
  }
  uint64_t current_pos() const override { return getProxiedOS().tell(); }
  size_t preferred_buffer_size() const override {
    return getPreferredBufferSize();
  }

public:
  void reserveExtraSpace(uint64_t ExtraSize) override {
    getProxiedOS().reserveExtraSpace(ExtraSize);
  }
  bool is_displayed() const override { return getProxiedOS().is_displayed(); }
  bool has_colors() const override { return getProxiedOS().has_colors(); }
  void enable_colors(bool enable) override {
    RawOstreamT::enable_colors(enable);
    getProxiedOS().enable_colors(enable);
  }

  ~raw_ostream_proxy_adaptor() override { resetProxiedOS(); }

protected:
  template <class... ArgsT>
  explicit raw_ostream_proxy_adaptor(raw_ostream &OS, ArgsT &&...Args)
      : RawOstreamT(std::forward<ArgsT>(Args)...),
        raw_ostream_proxy_adaptor_base(OS) {}

  /// Stop proxying the stream. Flush and set up a crash for future writes.
  ///
  /// For example, this can simplify logic when a subclass might have a longer
  /// lifetime than the stream it proxies.
  void resetProxiedOS() {
    raw_ostream_proxy_adaptor_base::resetProxiedOS(*this);
  }
  void resetProxiedOS(raw_ostream &) = delete;
};

/// Adaptor for creating a stream that proxies a \a raw_pwrite_stream.
template <class RawPwriteStreamT = raw_pwrite_stream>
class raw_pwrite_stream_proxy_adaptor
    : public raw_ostream_proxy_adaptor<RawPwriteStreamT> {
  using RawOstreamAdaptorT = raw_ostream_proxy_adaptor<RawPwriteStreamT>;

  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override {
    this->flush();
    getProxiedOS().pwrite(Ptr, Size, Offset);
  }

protected:
  raw_pwrite_stream_proxy_adaptor() = default;
  template <class... ArgsT>
  explicit raw_pwrite_stream_proxy_adaptor(raw_pwrite_stream &OS,
                                           ArgsT &&...Args)
      : RawOstreamAdaptorT(OS, std::forward<ArgsT>(Args)...) {}

  raw_pwrite_stream &getProxiedOS() const {
    return static_cast<raw_pwrite_stream &>(RawOstreamAdaptorT::getProxiedOS());
  }
};

/// Non-owning proxy for a \a raw_ostream. Enables passing a stream into of an
/// API that takes ownership.
class raw_ostream_proxy : public raw_ostream_proxy_adaptor<> {
  void anchor() override;

public:
  raw_ostream_proxy(raw_ostream &OS) : raw_ostream_proxy_adaptor<>(OS) {}
};

/// Non-owning proxy for a \a raw_pwrite_stream. Enables passing a stream
/// into of an API that takes ownership.
class raw_pwrite_stream_proxy : public raw_pwrite_stream_proxy_adaptor<> {
  void anchor() override;

public:
  raw_pwrite_stream_proxy(raw_pwrite_stream &OS)
      : raw_pwrite_stream_proxy_adaptor<>(OS) {}
};

} // end namespace llvm

#endif // LLVM_SUPPORT_RAW_OSTREAM_PROXY_H
