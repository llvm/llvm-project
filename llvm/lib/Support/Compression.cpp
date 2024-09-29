//===--- Compression.cpp - Compression implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements compression functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compression.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#if LLVM_ENABLE_ZLIB
#include <zlib.h>
#endif
#if LLVM_ENABLE_ZSTD
#include <zstd.h>
#endif

using namespace llvm;
using namespace llvm::compression;

const char *compression::getReasonIfUnsupported(compression::Format F) {
  switch (F) {
  case compression::Format::Zlib:
    if (zlib::isAvailable())
      return nullptr;
    return "LLVM was not built with LLVM_ENABLE_ZLIB or did not find zlib at "
           "build time";
  case compression::Format::Zstd:
    if (zstd::isAvailable())
      return nullptr;
    return "LLVM was not built with LLVM_ENABLE_ZSTD or did not find zstd at "
           "build time";
  }
  llvm_unreachable("");
}

void compression::compress(Params P, ArrayRef<uint8_t> Input,
                           SmallVectorImpl<uint8_t> &Output) {
  switch (P.format) {
  case compression::Format::Zlib:
    zlib::compress(Input, Output, P.level);
    break;
  case compression::Format::Zstd:
    zstd::compress(Input, Output, P.level, P.zstdEnableLdm);
    break;
  }
}

void compression::compressToStream(Params P, ArrayRef<uint8_t> Input,
                                   raw_ostream &OS) {
  switch (P.format) {
  case compression::Format::Zlib:
    zlib::compressToStream(Input, OS, P.level);
    break;
  case compression::Format::Zstd:
    zstd::compressToStream(Input, OS, P.level, P.zstdEnableLdm);
    break;
  }
}

Error compression::decompress(DebugCompressionType T, ArrayRef<uint8_t> Input,
                              uint8_t *Output, size_t UncompressedSize) {
  switch (formatFor(T)) {
  case compression::Format::Zlib:
    return zlib::decompress(Input, Output, UncompressedSize);
  case compression::Format::Zstd:
    return zstd::decompress(Input, Output, UncompressedSize);
  }
  llvm_unreachable("");
}

Error compression::decompress(compression::Format F, ArrayRef<uint8_t> Input,
                              SmallVectorImpl<uint8_t> &Output,
                              size_t UncompressedSize) {
  switch (F) {
  case compression::Format::Zlib:
    return zlib::decompress(Input, Output, UncompressedSize);
  case compression::Format::Zstd:
    return zstd::decompress(Input, Output, UncompressedSize);
  }
  llvm_unreachable("");
}

Error compression::decompress(DebugCompressionType T, ArrayRef<uint8_t> Input,
                              SmallVectorImpl<uint8_t> &Output,
                              size_t UncompressedSize) {
  return decompress(formatFor(T), Input, Output, UncompressedSize);
}

#if LLVM_ENABLE_ZLIB

static StringRef convertZlibCodeToString(int Code) {
  switch (Code) {
  case Z_MEM_ERROR:
    return "zlib error: Z_MEM_ERROR";
  case Z_BUF_ERROR:
    return "zlib error: Z_BUF_ERROR";
  case Z_STREAM_ERROR:
    return "zlib error: Z_STREAM_ERROR";
  case Z_DATA_ERROR:
    return "zlib error: Z_DATA_ERROR";
  case Z_OK:
  default:
    llvm_unreachable("unknown or unexpected zlib status code");
  }
}

bool zlib::isAvailable() { return true; }

void zlib::compress(ArrayRef<uint8_t> Input,
                    SmallVectorImpl<uint8_t> &CompressedBuffer, int Level) {
  unsigned long CompressedSize = ::compressBound(Input.size());
  CompressedBuffer.resize_for_overwrite(CompressedSize);
  int Res = ::compress2((Bytef *)CompressedBuffer.data(), &CompressedSize,
                        (const Bytef *)Input.data(), Input.size(), Level);
  if (Res == Z_MEM_ERROR)
    report_bad_alloc_error("Allocation failed");
  assert(Res == Z_OK);
  // Tell MemorySanitizer that zlib output buffer is fully initialized.
  // This avoids a false report when running LLVM with uninstrumented ZLib.
  __msan_unpoison(CompressedBuffer.data(), CompressedSize);
  if (CompressedSize < CompressedBuffer.size())
    CompressedBuffer.truncate(CompressedSize);
}

void zlib::compressToStream(ArrayRef<uint8_t> Input, raw_ostream &OS,
                            int Level) {
  // Allocate a fixed size buffer to hold the output.
  constexpr size_t OutBufferSize = 4096;
  auto OutBuffer = std::make_unique<Bytef[]>(OutBufferSize);

  z_stream ZStream;
  ZStream.zalloc = Z_NULL;
  ZStream.zfree = Z_NULL;
  ZStream.opaque = Z_NULL;

  int ZErr = deflateInit(&ZStream, Level);
  if (ZErr != Z_OK)
    report_bad_alloc_error("Failed to create ZStream");

  // Ensure that the z_stream is cleaned up on all exit paths.
  auto DeflateEndOnExit = make_scope_exit([&]() { deflateEnd(&ZStream); });

  ZStream.next_in =
      reinterpret_cast<Bytef *>(const_cast<uint8_t *>(Input.data()));
  ZStream.avail_in = Input.size();

  // Repeatedly deflate into the output buffer and flush it into the
  // output stream. Repeat until we have drained the entire compression
  // state.
  while (ZErr != Z_STREAM_END) {
    ZStream.next_out = OutBuffer.get();
    ZStream.avail_out = OutBufferSize;

    ZErr = deflate(&ZStream, Z_FINISH);
    if (ZErr == Z_STREAM_ERROR || ZErr == Z_BUF_ERROR)
      report_fatal_error(convertZlibCodeToString(ZErr));

    // Tell MemorySanitizer that zlib output buffer is fully initialized.
    // This avoids a false report when running LLVM with uninstrumented ZLib.
    __msan_unpoison(OutputBuffer.data(), OutBufferSize - ZStream.avail_out);

    if (ZStream.avail_out < OutBufferSize)
      OS.write(reinterpret_cast<char *>(OutBuffer.get()),
               OutBufferSize - ZStream.avail_out);
  }
}

Error zlib::decompress(ArrayRef<uint8_t> Input, uint8_t *Output,
                       size_t &UncompressedSize) {
  int Res = ::uncompress((Bytef *)Output, (uLongf *)&UncompressedSize,
                         (const Bytef *)Input.data(), Input.size());
  // Tell MemorySanitizer that zlib output buffer is fully initialized.
  // This avoids a false report when running LLVM with uninstrumented ZLib.
  __msan_unpoison(Output, UncompressedSize);
  return Res ? make_error<StringError>(convertZlibCodeToString(Res),
                                       inconvertibleErrorCode())
             : Error::success();
}

Error zlib::decompress(ArrayRef<uint8_t> Input,
                       SmallVectorImpl<uint8_t> &Output,
                       size_t UncompressedSize) {
  Output.resize_for_overwrite(UncompressedSize);
  Error E = zlib::decompress(Input, Output.data(), UncompressedSize);
  if (UncompressedSize < Output.size())
    Output.truncate(UncompressedSize);
  return E;
}

#else
bool zlib::isAvailable() { return false; }
void zlib::compress(ArrayRef<uint8_t> Input,
                    SmallVectorImpl<uint8_t> &CompressedBuffer, int Level) {
  llvm_unreachable("zlib::compress is unavailable");
}
void zlib::compressToStream(ArrayRef<uint8_t> Input, raw_ostream &OS,
                            int Level = DefaultCompression) {
  llvm_unreachable("zlib::compressToStream is unavailable");
}
Error zlib::decompress(ArrayRef<uint8_t> Input, uint8_t *UncompressedBuffer,
                       size_t &UncompressedSize) {
  llvm_unreachable("zlib::decompress is unavailable");
}
Error zlib::decompress(ArrayRef<uint8_t> Input,
                       SmallVectorImpl<uint8_t> &UncompressedBuffer,
                       size_t UncompressedSize) {
  llvm_unreachable("zlib::decompress is unavailable");
}
#endif

#if LLVM_ENABLE_ZSTD

bool zstd::isAvailable() { return true; }

#include <zstd.h> // Ensure ZSTD library is included

void zstd::compress(ArrayRef<uint8_t> Input,
                    SmallVectorImpl<uint8_t> &CompressedBuffer, int Level,
                    bool EnableLdm) {
  ZSTD_CCtx *Cctx = ZSTD_createCCtx();
  if (!Cctx)
    report_bad_alloc_error("Failed to create ZSTD_CCtx");

  if (ZSTD_isError(ZSTD_CCtx_setParameter(
          Cctx, ZSTD_c_enableLongDistanceMatching, EnableLdm ? 1 : 0))) {
    ZSTD_freeCCtx(Cctx);
    report_bad_alloc_error("Failed to set ZSTD_c_enableLongDistanceMatching");
  }

  if (ZSTD_isError(
          ZSTD_CCtx_setParameter(Cctx, ZSTD_c_compressionLevel, Level))) {
    ZSTD_freeCCtx(Cctx);
    report_bad_alloc_error("Failed to set ZSTD_c_compressionLevel");
  }

  unsigned long CompressedBufferSize = ZSTD_compressBound(Input.size());
  CompressedBuffer.resize_for_overwrite(CompressedBufferSize);

  size_t const CompressedSize =
      ZSTD_compress2(Cctx, CompressedBuffer.data(), CompressedBufferSize,
                     Input.data(), Input.size());

  ZSTD_freeCCtx(Cctx);

  if (ZSTD_isError(CompressedSize))
    report_bad_alloc_error("Compression failed");

  __msan_unpoison(CompressedBuffer.data(), CompressedSize);
  if (CompressedSize < CompressedBuffer.size())
    CompressedBuffer.truncate(CompressedSize);
}

void zstd::compressToStream(ArrayRef<uint8_t> Input, raw_ostream &OS, int Level,
                            bool EnableLdm) {
  // Allocate a buffer to hold the output.
  size_t OutBufferSize = ZSTD_CStreamOutSize();
  auto OutBuffer = std::make_unique<char[]>(OutBufferSize);

  ZSTD_CStream *CStream = ZSTD_createCStream();
  if (!CStream)
    report_bad_alloc_error("Failed to create ZSTD_CCtx");

  // Ensure that the ZSTD_CStream is cleaned up on all exit paths.
  auto FreeCStreamOnExit =
      make_scope_exit([=]() { ZSTD_freeCStream(CStream); });

  if (ZSTD_isError(ZSTD_CCtx_setParameter(
          CStream, ZSTD_c_enableLongDistanceMatching, EnableLdm ? 1 : 0))) {
    report_bad_alloc_error("Failed to set ZSTD_c_enableLongDistanceMatching");
  }

  if (ZSTD_isError(
          ZSTD_CCtx_setParameter(CStream, ZSTD_c_compressionLevel, Level))) {
    report_bad_alloc_error("Failed to set ZSTD_c_compressionLevel");
  }

  ZSTD_inBuffer ZInput = {Input.data(), Input.size(), 0};

  // Repeatedly compress into the output buffer and flush it into the
  // output stream. Repeat until we have drained the entire compression
  // state.
  size_t ZRet;
  do {
    ZSTD_outBuffer ZOutput = {OutBuffer.get(), OutBufferSize, 0};
    ZRet = ZSTD_compressStream2(CStream, &ZOutput, &ZInput, ZSTD_e_end);
    if (ZSTD_isError(ZRet))
      report_fatal_error(ZSTD_getErrorName(ZRet));

    // Tell MemorySanitizer that zstd output buffer is fully initialized.
    // This avoids a false report when running LLVM with uninstrumented ZStd.
    __msan_unpoison(OutputBuffer.data(), ZOutput.pos);

    if (ZOutput.pos > 0)
      OS.write(reinterpret_cast<char *>(OutBuffer.get()), ZOutput.pos);
  } while (ZRet != 0);
}

Error zstd::decompress(ArrayRef<uint8_t> Input, uint8_t *Output,
                       size_t &UncompressedSize) {
  const size_t Res = ::ZSTD_decompress(
      Output, UncompressedSize, (const uint8_t *)Input.data(), Input.size());
  UncompressedSize = Res;
  // Tell MemorySanitizer that zstd output buffer is fully initialized.
  // This avoids a false report when running LLVM with uninstrumented ZLib.
  __msan_unpoison(Output, UncompressedSize);
  return ZSTD_isError(Res) ? make_error<StringError>(ZSTD_getErrorName(Res),
                                                     inconvertibleErrorCode())
                           : Error::success();
}

Error zstd::decompress(ArrayRef<uint8_t> Input,
                       SmallVectorImpl<uint8_t> &Output,
                       size_t UncompressedSize) {
  Output.resize_for_overwrite(UncompressedSize);
  Error E = zstd::decompress(Input, Output.data(), UncompressedSize);
  if (UncompressedSize < Output.size())
    Output.truncate(UncompressedSize);
  return E;
}

#else
bool zstd::isAvailable() { return false; }
void zstd::compress(ArrayRef<uint8_t> Input,
                    SmallVectorImpl<uint8_t> &CompressedBuffer, int Level,
                    bool EnableLdm) {
  llvm_unreachable("zstd::compress is unavailable");
}
void zstd::compressToStream(ArrayRef<uint8_t> Input, raw_ostream &OS,
                            int Level = DefaultCompression,
                            bool EnableLdm = false) {
  llvm_unreachable("zstd::compressToStream is unavailable");
}
Error zstd::decompress(ArrayRef<uint8_t> Input, uint8_t *Output,
                       size_t &UncompressedSize) {
  llvm_unreachable("zstd::decompress is unavailable");
}
Error zstd::decompress(ArrayRef<uint8_t> Input,
                       SmallVectorImpl<uint8_t> &Output,
                       size_t UncompressedSize) {
  llvm_unreachable("zstd::decompress is unavailable");
}
#endif
