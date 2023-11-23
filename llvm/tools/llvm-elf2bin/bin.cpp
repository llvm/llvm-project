//===- bin.cpp - Code to write binary and VHX output for llvm-elf2bin -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-elf2bin.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <queue>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/*
 * Abstraction that provides a means of getting binary data from
 * somewhere. In the simple case this will involve reading data from a
 * segment in an ELF file. In more complex cases there might also be
 * zero-byte padding, or one of these stream objects filtering out the
 * even-index bytes of another.
 */
class BinaryDataStream {
public:
  // Returns a string of data, in whatever size is convenient. (But
  // the size should be bounded, so that other streams filtering this
  // one don't have to swallow a whole file in one go.)
  //
  // An empty returned string means EOF.
  virtual StringRef read() = 0;
  virtual ~BinaryDataStream() = default;
};

/*
 * BinaryDataStream implementation that reads data from an ELF image.
 */
class ElfSegment : public BinaryDataStream {
  InputObject &inobj;
  size_t position, remaining;

  StringRef read() override {
    size_t readlen = std::min<size_t>(remaining, 65536);
    size_t readpos = position;
    position += readlen;
    remaining -= readlen;
    return inobj.membuf->getBuffer().substr(readpos, readlen);
  }

public:
  ElfSegment(InputObject &inobj, size_t offset, size_t size)
      : inobj(inobj), position(offset), remaining(size) {}
};

/*
 * BinaryDataStream implementation that generates zero padding.
 */
class Padding : public BinaryDataStream {
  size_t remaining;
  char buffer[65536];

  StringRef read() override {
    size_t readlen = std::min<size_t>(remaining, 65536);
    remaining -= readlen;
    return StringRef(buffer, readlen);
  }

public:
  Padding(size_t size) : remaining(size) { memset(buffer, 0, sizeof(buffer)); }
};

/*
 * BinaryDataStream implementation that chains other BinaryDataStreams
 * together.
 */
class Chain : public BinaryDataStream {
  std::queue<std::unique_ptr<BinaryDataStream>> queue;

  StringRef read() override {
    while (!queue.empty()) {
      StringRef data = queue.front()->read();
      if (!data.empty())
        return data;

      queue.pop();
    }

    // If we get here, everything in our queue has finished.
    return "";
  }

public:
  Chain() = default;
  void push(std::unique_ptr<BinaryDataStream> &&st) {
    queue.push(std::move(st));
  }
};

/*
 * BinaryDataStream implementation that filters the output of another
 * BinaryDataStream so as to return only a subset of the bytes,
 * defined by a modulus and a range of residues. Specifically, a range
 * of 'nres' consecutive bytes from the underlying stream is passed
 * on, beginning at each byte whose index is congruent to 'firstres'
 * mod 'modulus' (regarding the first byte of the underlying stream as
 * having index 0).
 *
 * 'modulus' has to be a power of 2.
 */
class ModFilter : public BinaryDataStream {
  std::unique_ptr<BinaryDataStream> st;
  // Internal representation: 'mask' is the bitmask that reduces mod
  // 'modulus'. 'pos' iterates cyclically over 0,...,modulus-1 with
  // each byte we consume, and we return the byte if pos < nres.
  uint64_t mask, nres, pos;
  std::string outstring;

  StringRef read() override {
    outstring.clear();
    llvm::raw_string_ostream outstream(outstring);

    while (true) {
      StringRef data = st->read();
      if (data.empty())
        break;

      for (char c : data) {
        if (pos < nres)
          outstream << c;
        pos = (pos + 1) & mask;
      }

      // If that batch of input contributed no bytes to our
      // output, go round again. Otherwise, we have something to
      // return.
      if (!outstring.empty())
        break;
    }
    return outstring;
  }

public:
  ModFilter(std::unique_ptr<BinaryDataStream> &&st, uint64_t modulus,
            uint64_t firstres, uint64_t nres)
      : st(std::move(st)), nres(nres) {
    // Check input values are reasonable.
    assert(llvm::has_single_bit(modulus));
    assert(nres > 0);
    assert(nres < modulus);

    mask = modulus - 1;

    // Set pos to be (-firstres), so that we'll skip the right
    // number of bytes before the first one we return.
    //
    // Written as (1 + ~firstres) to avoid Visual Studio
    // complaining about negating an unsigned.
    pos = (1 + ~firstres) & mask;
  }
};

static void bin_write(BinaryDataStream &st, const std::string &outfile) {
  std::error_code error;
  llvm::raw_fd_ostream ofs(outfile, error);
  if (error)
    fatal(outfile, "unable to open", errorCodeToError(error));

  while (true) {
    StringRef data = st.read();
    if (data.empty())
      return;
    ofs << data;
  }
}

static void vhx_write(BinaryDataStream &st, const std::string &outfile) {
  std::error_code error;
  llvm::raw_fd_ostream ofs(outfile, error);
  if (error)
    fatal(outfile, "unable to open", errorCodeToError(error));

  while (true) {
    StringRef data = st.read();
    if (data.empty())
      return;
    for (uint8_t c : data) {
      static const char hexdigits[] = "0123456789ABCDEF";
      ofs << hexdigits[c >> 4] << hexdigits[c & 0xF] << '\n';
    }
  }
}

static std::unique_ptr<BinaryDataStream> onesegment_prepare(InputObject &inobj,
                                                            uint64_t fileoffset,
                                                            uint64_t size,
                                                            uint64_t zi_size) {
  auto base_stream = std::make_unique<ElfSegment>(inobj, fileoffset, size);

  if (!zi_size)
    return base_stream;

  auto chain = std::make_unique<Chain>();
  chain->push(std::move(base_stream));
  chain->push(std::make_unique<Padding>(zi_size));
  return chain;
}

static std::unique_ptr<BinaryDataStream>
combined_prepare(InputObject &inobj, const std::vector<Segment> &segments_orig,
                 bool include_zi, std::optional<uint64_t> baseaddr) {
  // Sort the segments by base address, in case they weren't already.
  struct {
    bool operator()(const Segment &a, const Segment &b) {
      return a.baseaddr < b.baseaddr;
    }
  } comparator;
  std::vector<Segment> sorted = segments_orig;
  std::sort(sorted.begin(), sorted.end(), comparator);

  // Spot and reject overlapping segments.
  //
  // (WIBNI: we _could_ tolerate these if they also agreed on what
  // part of the ELF file corresponded to the overlapping range of
  // address space. I don't see a reason to implement that in
  // advance of someone actually having a good use for it, but
  // that's why I'm leaving this overlap check as a separate pass
  // rather than folding it into the next one - this way, we could
  // write a modified set of segments into 'nonoverlapping'.)
  std::vector<Segment> nonoverlapping;
  if (!sorted.empty()) {
    auto it = sorted.begin(), end = sorted.end();

    if (baseaddr && it->baseaddr < baseaddr.value())
      fatal(inobj, Twine("first segment is at address 0x") +
                       Twine::utohexstr(it->baseaddr) +
                       ", below the specified base address 0x" +
                       Twine::utohexstr(baseaddr.value()));

    nonoverlapping.push_back(*it++);
    for (; it != end; ++it) {
      const auto &prev = nonoverlapping.back(), curr = *it;
      if (curr.baseaddr - prev.baseaddr < prev.memsize)
        fatal(inobj, Twine("segments at addresses 0x")
              + Twine::utohexstr(prev.baseaddr) + " and 0x"
              + Twine::utohexstr(curr.baseaddr) + " overlap");
      nonoverlapping.push_back(curr);
    }
  }

  // Make a chained output stream that inserts the right padding
  // between all those segments.
  auto chain = std::make_unique<Chain>();
  if (!nonoverlapping.empty()) {
    uint64_t addr =
        (baseaddr ? baseaddr.value() : nonoverlapping.begin()->baseaddr);

    for (const auto &seg : nonoverlapping) {
      if (addr < seg.baseaddr)
        chain->push(std::make_unique<Padding>(seg.baseaddr - addr));
      chain->push(
          std::make_unique<ElfSegment>(inobj, seg.fileoffset, seg.filesize));
      addr = seg.baseaddr + seg.filesize;

      if (include_zi && seg.memsize > seg.filesize) {
        chain->push(std::make_unique<Padding>(seg.memsize - seg.filesize));
        addr = seg.baseaddr + seg.memsize;
      }
    }
  }
  return chain;
}

static std::unique_ptr<BinaryDataStream>
bank_prepare(std::unique_ptr<BinaryDataStream> stream, uint64_t bank_modulus,
             uint64_t bank_firstres, uint64_t bank_nres) {
  if (bank_modulus == 1)
    return stream;

  return std::make_unique<ModFilter>(std::move(stream), bank_modulus,
                                     bank_firstres, bank_nres);
}

void bin_write(InputObject &inobj, const std::string &outfile,
               uint64_t fileoffset, uint64_t size, uint64_t zi_size,
               uint64_t bank_modulus, uint64_t bank_firstres,
               uint64_t bank_nres) {
  auto streamp = onesegment_prepare(inobj, fileoffset, size, zi_size);
  streamp =
      bank_prepare(std::move(streamp), bank_modulus, bank_firstres, bank_nres);
  bin_write(*streamp, outfile);
}

void bincombined_write(InputObject &inobj, const std::string &outfile,
                       const std::vector<Segment> &segments, bool include_zi,
                       std::optional<uint64_t> baseaddr, uint64_t bank_modulus,
                       uint64_t bank_firstres, uint64_t bank_nres) {
  auto streamp = combined_prepare(inobj, segments, include_zi, baseaddr);
  streamp =
      bank_prepare(std::move(streamp), bank_modulus, bank_firstres, bank_nres);
  bin_write(*streamp, outfile);
}

void vhx_write(InputObject &inobj, const std::string &outfile,
               uint64_t fileoffset, uint64_t size, uint64_t zi_size,
               uint64_t bank_modulus, uint64_t bank_firstres,
               uint64_t bank_nres) {
  auto streamp = onesegment_prepare(inobj, fileoffset, size, zi_size);
  streamp =
      bank_prepare(std::move(streamp), bank_modulus, bank_firstres, bank_nres);
  vhx_write(*streamp, outfile);
}

void vhxcombined_write(InputObject &inobj, const std::string &outfile,
                       const std::vector<Segment> &segments, bool include_zi,
                       std::optional<uint64_t> baseaddr, uint64_t bank_modulus,
                       uint64_t bank_firstres, uint64_t bank_nres) {
  auto streamp = combined_prepare(inobj, segments, include_zi, baseaddr);
  streamp =
      bank_prepare(std::move(streamp), bank_modulus, bank_firstres, bank_nres);
  vhx_write(*streamp, outfile);
}
