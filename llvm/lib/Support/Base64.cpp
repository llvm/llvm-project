//===- Base64.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define INVALID_BASE64_BYTE 64
#include "llvm/Support/Base64.h"

static char decodeBase64Byte(uint8_t Ch) {
  constexpr char Inv = INVALID_BASE64_BYTE;
  static const char DecodeTable[] = {
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, 62,  Inv, Inv, Inv, 63,  // ...+.../
      52,  53,  54,  55,  56,  57,  58,  59,  // 01234567
      60,  61,  Inv, Inv, Inv, 0,   Inv, Inv, // 89...=..
      Inv, 0,   1,   2,   3,   4,   5,   6,   // .ABCDEFG
      7,   8,   9,   10,  11,  12,  13,  14,  // HIJKLMNO
      15,  16,  17,  18,  19,  20,  21,  22,  // PQRSTUVW
      23,  24,  25,  Inv, Inv, Inv, Inv, Inv, // XYZ.....
      Inv, 26,  27,  28,  29,  30,  31,  32,  // .abcdefg
      33,  34,  35,  36,  37,  38,  39,  40,  // hijklmno
      41,  42,  43,  44,  45,  46,  47,  48,  // pqrstuvw
      49,  50,  51                            // xyz.....
  };
  if (Ch >= sizeof(DecodeTable))
    return Inv;
  return DecodeTable[Ch];
}

llvm::Error llvm::decodeBase64(llvm::StringRef Input,
                               std::vector<char> &Output) {
  constexpr char Base64InvalidByte = INVALID_BASE64_BYTE;
  // Invalid table value with short name to fit in the table init below. The
  // invalid value is 64 since valid base64 values are 0 - 63.
  Output.clear();
  const uint64_t InputLength = Input.size();
  if (InputLength == 0)
    return Error::success();
  // Make sure we have a valid input string length which must be a multiple
  // of 4.
  if ((InputLength % 4) != 0)
    return createStringError(std::errc::illegal_byte_sequence,
                             "Base64 encoded strings must be a multiple of 4 "
                             "bytes in length");
  const uint64_t FirstValidEqualIdx = InputLength - 2;
  char Hex64Bytes[4];
  for (uint64_t Idx = 0; Idx < InputLength; Idx += 4) {
    for (uint64_t ByteOffset = 0; ByteOffset < 4; ++ByteOffset) {
      const uint64_t ByteIdx = Idx + ByteOffset;
      const char Byte = Input[ByteIdx];
      const char DecodedByte = decodeBase64Byte(Byte);
      bool Illegal = DecodedByte == Base64InvalidByte;
      if (!Illegal && Byte == '=') {
        if (ByteIdx < FirstValidEqualIdx) {
          // We have an '=' in the middle of the string which is invalid, only
          // the last two characters can be '=' characters.
          Illegal = true;
        } else if (ByteIdx == FirstValidEqualIdx && Input[ByteIdx + 1] != '=') {
          // We have an equal second to last from the end and the last character
          // is not also an equal, so the '=' character is invalid
          Illegal = true;
        }
      }
      if (Illegal)
        return createStringError(
            std::errc::illegal_byte_sequence,
            "Invalid Base64 character %#2.2x at index %" PRIu64, Byte, ByteIdx);
      Hex64Bytes[ByteOffset] = DecodedByte;
    }
    // Now we have 6 bits of 3 bytes in value in each of the Hex64Bytes bytes.
    // Extract the right bytes into the Output buffer.
    Output.push_back((Hex64Bytes[0] << 2) + ((Hex64Bytes[1] >> 4) & 0x03));
    Output.push_back((Hex64Bytes[1] << 4) + ((Hex64Bytes[2] >> 2) & 0x0f));
    Output.push_back((Hex64Bytes[2] << 6) + (Hex64Bytes[3] & 0x3f));
  }
  // If we had valid trailing '=' characters strip the right number of bytes
  // from the end of the output buffer. We already know that the Input length
  // it a multiple of 4 and is not zero, so direct character access is safe.
  if (Input.back() == '=') {
    Output.pop_back();
    if (Input[InputLength - 2] == '=')
      Output.pop_back();
  }
  return Error::success();
}

using namespace llvm;

namespace {

using byte = std::byte;

::llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

class Base64Impl {
private:
  static constexpr char EncodingTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "abcdefghijklmnopqrstuvwxyz"
                                          "0123456789+/";

  static_assert(sizeof(EncodingTable) == 65, "");

  // Compose an index into the encoder table from two bytes and the number of
  // significant bits in the lower byte until the byte boundary.
  static inline int composeInd(byte ByteLo, byte ByteHi, int BitsLo) {
    int Res = (int)((ByteHi << BitsLo) | (ByteLo >> (8 - BitsLo))) & 0x3F;
    return Res;
  }

  // Decode a single character.
  static inline int decode(char Ch) {
    if (Ch >= 'A' && Ch <= 'Z') // 0..25
      return Ch - 'A';
    else if (Ch >= 'a' && Ch <= 'z') // 26..51
      return Ch - 'a' + 26;
    else if (Ch >= '0' && Ch <= '9') // 52..61
      return Ch - '0' + 52;
    else if (Ch == '+') // 62
      return 62;
    else if (Ch == '/') // 63
      return 63;
    return -1;
  }

  // Decode a quadruple of characters.
  static inline Expected<bool> decode4(const char *Src, byte *Dst) {
    int BadCh = -1;

    for (auto I = 0; I < 4; ++I) {
      char Ch = Src[I];
      int Byte = decode(Ch);

      if (Byte < 0) {
        BadCh = Ch;
        break;
      }
      Dst[I] = (byte)Byte;
    }
    if (BadCh == -1)
      return true;
    return makeError("invalid char in Base64Impl encoding: 0x" + Twine(BadCh));
  }

public:
  static size_t getEncodedSize(size_t SrcSize) {
    constexpr int ByteSizeInBits = 8;
    constexpr int EncBitsPerChar = 6;
    return (SrcSize * ByteSizeInBits + (EncBitsPerChar - 1)) / EncBitsPerChar;
  }

  static size_t encode(const byte *Src, raw_ostream &Out, size_t SrcSize) {
    size_t Off = 0;

    // encode full byte triples
    for (size_t TriB = 0; TriB < SrcSize / 3; ++TriB) {
      Off = TriB * 3;
      byte Byte0 = Src[Off++];
      byte Byte1 = Src[Off++];
      byte Byte2 = Src[Off++];

      Out << EncodingTable[(int)Byte0 & 0x3F];
      Out << EncodingTable[composeInd(Byte0, Byte1, 2)];
      Out << EncodingTable[composeInd(Byte1, Byte2, 4)];
      Out << EncodingTable[(int)(Byte2 >> 2) & 0x3F];
    }
    // encode the remainder
    int RemBytes = SrcSize - Off;

    if (RemBytes > 0) {
      byte Byte0 = Src[Off + 0];
      Out << EncodingTable[(int)Byte0 & 0x3F];

      if (RemBytes > 1) {
        byte Byte1 = Src[Off + 1];
        Out << EncodingTable[composeInd(Byte0, Byte1, 2)];
        Out << EncodingTable[(int)(Byte1 >> 4) & 0x3F];
      } else {
        Out << EncodingTable[(int)(Byte0 >> 6) & 0x3F];
      }
    }
    return getEncodedSize(SrcSize);
  }

  static size_t getDecodedSize(size_t SrcSize) { return (SrcSize * 3 + 3) / 4; }

  static Expected<size_t> decode(const char *Src, byte *Dst, size_t SrcSize) {
    size_t SrcOff = 0;
    size_t DstOff = 0;

    // decode full quads
    for (size_t Qch = 0; Qch < SrcSize / 4; ++Qch, SrcOff += 4, DstOff += 3) {
      byte Ch[4];
      Expected<bool> TrRes = decode4(Src + SrcOff, Ch);

      if (!TrRes)
        return TrRes.takeError();
      // each quad of chars produces three bytes of output
      Dst[DstOff + 0] = Ch[0] | (Ch[1] << 6);
      Dst[DstOff + 1] = (Ch[1] >> 2) | (Ch[2] << 4);
      Dst[DstOff + 2] = (Ch[2] >> 4) | (Ch[3] << 2);
    }
    auto RemChars = SrcSize - SrcOff;

    if (RemChars == 0)
      return DstOff;
    // decode the remainder; variants:
    // 2 chars remain - produces single byte
    // 3 chars remain - produces two bytes

    if (RemChars != 2 && RemChars != 3)
      return makeError("invalid encoded sequence length");

    int Ch0 = decode(Src[SrcOff++]);
    int Ch1 = decode(Src[SrcOff++]);
    int Ch2 = RemChars == 3 ? decode(Src[SrcOff]) : 0;

    if (Ch0 < 0 || Ch1 < 0 || Ch2 < 0)
      return makeError("invalid characters in the encoded sequence remainder");
    Dst[DstOff++] = (byte)Ch0 | (byte)((Ch1 << 6));

    if (RemChars == 3)
      Dst[DstOff++] = (byte)(Ch1 >> 2) | (byte)(Ch2 << 4);
    return DstOff;
  }

  static Expected<std::unique_ptr<byte[]>> decode(const char *Src,
                                                  size_t SrcSize) {
    size_t DstSize = getDecodedSize(SrcSize);
    std::unique_ptr<byte[]> Dst(new byte[DstSize]);
    Expected<size_t> Res = decode(Src, Dst.get(), SrcSize);
    if (!Res)
      return Res.takeError();
    return Expected<std::unique_ptr<byte[]>>(std::move(Dst));
  }
};

constexpr char Base64Impl::EncodingTable[];

} // anonymous namespace

size_t Base64::getEncodedSize(size_t SrcSize) {
  return Base64Impl::getEncodedSize(SrcSize);
}

size_t Base64::encode(const byte *Src, raw_ostream &Out, size_t SrcSize) {
  return Base64Impl::encode(Src, Out, SrcSize);
}

size_t Base64::getDecodedSize(size_t SrcSize) {
  return Base64Impl::getDecodedSize(SrcSize);
}

Expected<size_t> Base64::decode(const char *Src, byte *Dst, size_t SrcSize) {
  return Base64Impl::decode(Src, Dst, SrcSize);
}

Expected<std::unique_ptr<byte[]>> Base64::decode(const char *Src,
                                                 size_t SrcSize) {
  return Base64Impl::decode(Src, SrcSize);
}