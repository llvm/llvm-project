//===- hex.cpp - Code to write Intel and Motorola Hex for llvm-elf2bin ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-elf2bin.h"

#include <assert.h>

#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

class Hex {
public:
  virtual void data(uint64_t addr, const std::string &data) = 0;
  virtual void trailer(uint64_t entry) = 0;
  virtual ~Hex() = default;
};

template <typename Integer> std::string bigend(Integer i, size_t bytes_wanted) {
  assert(bytes_wanted <= sizeof(i));
  char buf[sizeof(i)];
  llvm::support::endian::write(buf, i, llvm::endianness::big);
  return std::string(buf + sizeof(i) - bytes_wanted, bytes_wanted);
}

class IHex : public Hex {
  static std::string record(uint8_t type, uint16_t addr,
                            const std::string &data) {
    std::string binstring;
    llvm::raw_string_ostream binstream(binstring);

    binstream << (char)data.size() << bigend(addr, 2) << (char)type << data;

    uint8_t checksum = 0;
    for (uint8_t c : binstring)
      checksum -= c;
    binstream << (char)checksum;

    std::string hexstring;
    llvm::raw_string_ostream hexstream(hexstring);

    hexstream << ':';
    for (uint8_t c : binstring) {
      static const char hexdigits[] = "0123456789ABCDEF";
      hexstream << hexdigits[c >> 4] << hexdigits[c & 0xF];
    }
    hexstream << '\n';
    return hexstring;
  }

  InputObject &inobj;
  llvm::raw_ostream &os;
  uint64_t curr_offset = 0;

  void data(uint64_t addr, const std::string &data) override {
    uint64_t offset = addr >> 16;
    if (offset != curr_offset) {
      if (offset >= 0x10000)
        fatal(inobj, "data address does not fit in 32 bits");
      curr_offset = offset;
      os << record(4, 0, bigend(curr_offset, 2));
    }
    os << record(0, addr & 0xFFFF, data);
  }

  void trailer(uint64_t entry) override {
    if (entry >= 0x100000000)
      fatal(inobj, "entry point does not fit in 32 bits");
    os << record(5, 0, bigend(entry, 4)); // entry point
    os << record(1, 0, "");               // EOF
  }

public:
  static constexpr uint64_t max_datalen = 0xFF;
  IHex(InputObject &inobj, llvm::raw_ostream &os) : inobj(inobj), os(os) {}
};

class SRec : public Hex {
  static std::string record(uint8_t type, uint32_t addr,
                            const std::string &data) {
    std::string binstring;
    llvm::raw_string_ostream binstream(binstring);

    size_t addrsize = (type == 2 || type == 8   ? 3
                       : type == 3 || type == 7 ? 4
                                                : 2);

    binstream << (char)(data.size() + addrsize + 1) << bigend(addr, addrsize)
              << data;

    uint8_t checksum = -1;
    for (uint8_t c : binstring)
      checksum -= c;
    binstream << (char)checksum;

    std::string hexstring;
    llvm::raw_string_ostream hexstream(hexstring);

    hexstream << 'S' << (char)('0' + type);
    for (uint8_t c : binstring) {
      static const char hexdigits[] = "0123456789ABCDEF";
      hexstream << hexdigits[c >> 4] << hexdigits[c & 0xF];
    }
    hexstream << '\n';
    return hexstring;
  }

  InputObject &inobj;
  llvm::raw_ostream &os;

  void data(uint64_t addr, const std::string &data) override {
    if (addr >= 0x100000000)
      fatal(inobj, "data address does not fit in 32 bits");
    os << record(3, static_cast<uint32_t>(addr), data);
  }

  void trailer(uint64_t entry) override {
    if (entry >= 0x100000000)
      fatal(inobj, "entry point does not fit in 32 bits");

    // We could also write an S5 or S6 record here, containing the total number
    // of data records in the file. However, srec_motorola(5) says one of these
    // is optional, and I'm unaware of anyone depending on it existing. Also,
    // we'd have to decide what to do if the file were so huge that the number
    // wouldn't fit.

    os << record(7, static_cast<uint32_t>(entry), "");
  }

public:
  // In S-records, the length field includes the address and checksum,
  // so we can have fewer data bytes in a record than 0xFF
  static constexpr uint64_t max_datalen = 0xFF - 5;

  SRec(InputObject &inobj, llvm::raw_ostream &os) : inobj(inobj), os(os) {}
};

static void hex_write(InputObject &inobj, Hex &hex,
                      const std::vector<Segment> &segments, bool include_zi,
                      uint64_t datareclen) {
  for (auto seg : segments) {
    uint64_t segsize = include_zi ? seg.memsize : seg.filesize;

    for (uint64_t pos = 0; pos < segsize; pos += datareclen) {
      size_t thisreclen = std::min<size_t>(datareclen, segsize - pos);
      size_t readlen = pos >= seg.filesize
                           ? 0
                           : std::min<size_t>(thisreclen, seg.filesize - pos);

      std::string data;
      if (readlen)
        data = std::string(
            inobj.membuf->getBuffer().substr(seg.fileoffset + pos, readlen));
      if (thisreclen > readlen)
        data += std::string(thisreclen - readlen, '\0');
      hex.data(seg.baseaddr + pos, data);
    }
  }

  hex.trailer(inobj.entry_point());
}

template <typename HexFormat>
static void hex_write(InputObject &inobj, const std::string &outfile,
                      const std::vector<Segment> &segments, bool include_zi,
                      uint64_t datareclen) {
  if (datareclen > HexFormat::max_datalen)
    fatal(inobj, "data record length must be at most " +
                     Twine(unsigned(HexFormat::max_datalen)));

  if (datareclen < 1)
    fatal(inobj, "data record length must be at least 1");

  std::error_code error;
  llvm::raw_fd_ostream outstream(outfile, error);
  if (error)
    fatal(outfile, "unable to open", errorCodeToError(error));

  HexFormat hex(inobj, outstream);
  hex_write(inobj, hex, segments, include_zi, datareclen);
}

void ihex_write(InputObject &inobj, const std::string &outfile,
                const std::vector<Segment> &segments, bool include_zi,
                uint64_t datareclen) {
  hex_write<IHex>(inobj, outfile, segments, include_zi, datareclen);
}

void srec_write(InputObject &inobj, const std::string &outfile,
                const std::vector<Segment> &segments, bool include_zi,
                uint64_t datareclen) {
  hex_write<SRec>(inobj, outfile, segments, include_zi, datareclen);
}
