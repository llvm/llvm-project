//===- llvm/unittest/DebugInfo/DWARFDebugFrameTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

dwarf::CIE createCIE(bool IsDWARF64, uint64_t Offset, uint64_t Length) {
  return dwarf::CIE(IsDWARF64, Offset, Length,
                    /*Version=*/3,
                    /*Augmentation=*/StringRef(),
                    /*AddressSize=*/8,
                    /*SegmentDescriptorSize=*/0,
                    /*CodeAlignmentFactor=*/1,
                    /*DataAlignmentFactor=*/-8,
                    /*ReturnAddressRegister=*/16,
                    /*AugmentationData=*/StringRef(),
                    /*FDEPointerEncoding=*/dwarf::DW_EH_PE_absptr,
                    /*LSDAPointerEncoding=*/dwarf::DW_EH_PE_omit,
                    /*Personality=*/None,
                    /*PersonalityEnc=*/None,
                    /*Arch=*/Triple::x86_64);
}

void expectDumpResult(const dwarf::CIE &TestCIE, bool IsEH,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  TestCIE.dump(OS, /*MRI=*/nullptr, IsEH);
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

void expectDumpResult(const dwarf::FDE &TestFDE, bool IsEH,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  TestFDE.dump(OS, /*MRI=*/nullptr, IsEH);
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

TEST(DWARFDebugFrame, DumpDWARF32CIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x1111abcd,
                                 /*Length=*/0x2222abcd);
  expectDumpResult(TestCIE, /*IsEH=*/false, "1111abcd 2222abcd ffffffff CIE");
}

TEST(DWARFDebugFrame, DumpDWARF64CIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1111abcdabcd,
                                 /*Length=*/0x2222abcdabcd);
  expectDumpResult(TestCIE, /*IsEH=*/false,
                   "1111abcdabcd 00002222abcdabcd ffffffffffffffff CIE");
}

TEST(DWARFDebugFrame, DumpEHCIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x1000,
                                 /*Length=*/0x20);
  expectDumpResult(TestCIE, /*IsEH=*/true, "00001000 00000020 00000000 CIE");
}

TEST(DWARFDebugFrame, DumpEH64CIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1000,
                                 /*Length=*/0x20);
  expectDumpResult(TestCIE, /*IsEH=*/true,
                   "00001000 0000000000000020 00000000 CIE");
}

TEST(DWARFDebugFrame, DumpDWARF64FDE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1111abcdabcd,
                                 /*Length=*/0x2222abcdabcd);
  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x5555abcdabcd,
                     /*AddressRange=*/0x111111111111,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/None,
                     /*Arch=*/Triple::x86_64);
  expectDumpResult(TestFDE, /*IsEH=*/false,
                   "3333abcdabcd 00004444abcdabcd 00001111abcdabcd FDE "
                   "cie=1111abcdabcd pc=5555abcdabcd...6666bcdebcde");
}

TEST(DWARFDebugFrame, DumpEH64FDE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1111ab9a000c,
                                 /*Length=*/0x20);
  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x1111abcdabcd,
                     /*Length=*/0x2222abcdabcd,
                     /*CIEPointer=*/0x33abcd,
                     /*InitialLocation=*/0x4444abcdabcd,
                     /*AddressRange=*/0x111111111111,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/None,
                     /*Arch=*/Triple::x86_64);
  expectDumpResult(TestFDE, /*IsEH=*/true,
                   "1111abcdabcd 00002222abcdabcd 0033abcd FDE "
                   "cie=1111ab9a000c pc=4444abcdabcd...5555bcdebcde");
}

} // end anonymous namespace
