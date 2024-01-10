//===-- RISCVFeatures.td - RISC-V Features and Extensions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

def FeatureStdExtZicsr
    : SubtargetFeature<"zicsr", "HasStdExtZicsr", "true",
                       "'zicsr' (CSRs)">;
def HasStdExtZicsr : Predicate<"Subtarget->hasStdExtZicsr()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicsr),
                                "'Zicsr' (CSRs)">;

def FeatureStdExtM
    : SubtargetFeature<"m", "HasStdExtM", "true",
                       "'M' (Integer Multiplication and Division)">;
def HasStdExtM : Predicate<"Subtarget->hasStdExtM()">,
                           AssemblerPredicate<(all_of FeatureStdExtM),
                           "'M' (Integer Multiplication and Division)">;

def FeatureStdExtZmmul
    : SubtargetFeature<"zmmul", "HasStdExtZmmul", "true",
                       "'Zmmul' (Integer Multiplication)">;

def HasStdExtMOrZmmul
    : Predicate<"Subtarget->hasStdExtM() || Subtarget->hasStdExtZmmul()">,
                AssemblerPredicate<(any_of FeatureStdExtM, FeatureStdExtZmmul),
                                   "'M' (Integer Multiplication and Division) or "
                                   "'Zmmul' (Integer Multiplication)">;

def FeatureStdExtA
    : SubtargetFeature<"a", "HasStdExtA", "true",
                       "'A' (Atomic Instructions)">;
def HasStdExtA : Predicate<"Subtarget->hasStdExtA()">,
                           AssemblerPredicate<(all_of FeatureStdExtA),
                           "'A' (Atomic Instructions)">;

def FeatureStdExtF
    : SubtargetFeature<"f", "HasStdExtF", "true",
                       "'F' (Single-Precision Floating-Point)",
                       [FeatureStdExtZicsr]>;
def HasStdExtF : Predicate<"Subtarget->hasStdExtF()">,
                           AssemblerPredicate<(all_of FeatureStdExtF),
                           "'F' (Single-Precision Floating-Point)">;

def FeatureStdExtD
    : SubtargetFeature<"d", "HasStdExtD", "true",
                       "'D' (Double-Precision Floating-Point)",
                       [FeatureStdExtF]>;
def HasStdExtD : Predicate<"Subtarget->hasStdExtD()">,
                           AssemblerPredicate<(all_of FeatureStdExtD),
                           "'D' (Double-Precision Floating-Point)">;

def FeatureStdExtH
    : SubtargetFeature<"h", "HasStdExtH", "true",
                       "'H' (Hypervisor)">;

def HasStdExtH : Predicate<"Subtarget->hasStdExtH()">,
                           AssemblerPredicate<(all_of FeatureStdExtH),
                           "'H' (Hypervisor)">;

def FeatureStdExtZihintpause
    : SubtargetFeature<"zihintpause", "HasStdExtZihintpause", "true",
                       "'Zihintpause' (Pause Hint)">;
def HasStdExtZihintpause : Predicate<"Subtarget->hasStdExtZihintpause()">,
                                     AssemblerPredicate<(all_of FeatureStdExtZihintpause),
                                     "'Zihintpause' (Pause Hint)">;

def FeatureStdExtZihintntl
    : SubtargetFeature<"zihintntl", "HasStdExtZihintntl", "true",
                       "'Zihintntl' (Non-Temporal Locality Hints)">;
def HasStdExtZihintntl : Predicate<"Subtarget->hasStdExtZihintntl()">,
                                    AssemblerPredicate<(all_of FeatureStdExtZihintntl),
                                    "'Zihintntl' (Non-Temporal Locality Hints)">;

def FeatureStdExtZifencei
    : SubtargetFeature<"zifencei", "HasStdExtZifencei", "true",
                       "'Zifencei' (fence.i)">;
def HasStdExtZifencei : Predicate<"Subtarget->hasStdExtZifencei()">,
                                   AssemblerPredicate<(all_of FeatureStdExtZifencei),
                                   "'Zifencei' (fence.i)">;

def FeatureStdExtZicntr
    : SubtargetFeature<"zicntr", "HasStdExtZicntr", "true",
                       "'Zicntr' (Base Counters and Timers)",
                       [FeatureStdExtZicsr]>;

def FeatureStdExtZihpm
    : SubtargetFeature<"zihpm", "HasStdExtZihpm", "true",
                       "'Zihpm' (Hardware Performance Counters)",
                       [FeatureStdExtZicsr]>;

def FeatureStdExtZfhmin
    : SubtargetFeature<"zfhmin", "HasStdExtZfhmin", "true",
                       "'Zfhmin' (Half-Precision Floating-Point Minimal)",
                       [FeatureStdExtF]>;
def HasStdExtZfhmin : Predicate<"Subtarget->hasStdExtZfhmin()">,
                             AssemblerPredicate<(all_of FeatureStdExtZfhmin),
                             "'Zfhmin' (Half-Precision Floating-Point Minimal)">;

def FeatureStdExtZfh
    : SubtargetFeature<"zfh", "HasStdExtZfh", "true",
                       "'Zfh' (Half-Precision Floating-Point)",
                       [FeatureStdExtZfhmin]>;
def HasStdExtZfh : Predicate<"Subtarget->hasStdExtZfh()">,
                             AssemblerPredicate<(all_of FeatureStdExtZfh),
                             "'Zfh' (Half-Precision Floating-Point)">;
def NoStdExtZfh : Predicate<"!Subtarget->hasStdExtZfh()">;

def HasStdExtZfhOrZfhmin
    : Predicate<"Subtarget->hasStdExtZfhmin()">,
                AssemblerPredicate<(all_of FeatureStdExtZfhmin),
                                   "'Zfh' (Half-Precision Floating-Point) or "
                                   "'Zfhmin' (Half-Precision Floating-Point Minimal)">;

def FeatureStdExtZfinx
    : SubtargetFeature<"zfinx", "HasStdExtZfinx", "true",
                       "'Zfinx' (Float in Integer)",
                       [FeatureStdExtZicsr]>;
def HasStdExtZfinx : Predicate<"Subtarget->hasStdExtZfinx()">,
                               AssemblerPredicate<(all_of FeatureStdExtZfinx),
                               "'Zfinx' (Float in Integer)">;

def FeatureStdExtZdinx
    : SubtargetFeature<"zdinx", "HasStdExtZdinx", "true",
                       "'Zdinx' (Double in Integer)",
                       [FeatureStdExtZfinx]>;
def HasStdExtZdinx : Predicate<"Subtarget->hasStdExtZdinx()">,
                               AssemblerPredicate<(all_of FeatureStdExtZdinx),
                               "'Zdinx' (Double in Integer)">;

def FeatureStdExtZhinxmin
    : SubtargetFeature<"zhinxmin", "HasStdExtZhinxmin", "true",
                       "'Zhinxmin' (Half Float in Integer Minimal)",
                       [FeatureStdExtZfinx]>;
def HasStdExtZhinxmin : Predicate<"Subtarget->hasStdExtZhinxmin()">,
                                  AssemblerPredicate<(all_of FeatureStdExtZhinxmin),
                                  "'Zhinxmin' (Half Float in Integer Minimal)">;

def FeatureStdExtZhinx
    : SubtargetFeature<"zhinx", "HasStdExtZhinx", "true",
                       "'Zhinx' (Half Float in Integer)",
                       [FeatureStdExtZhinxmin]>;
def HasStdExtZhinx : Predicate<"Subtarget->hasStdExtZhinx()">,
                               AssemblerPredicate<(all_of FeatureStdExtZhinx),
                               "'Zhinx' (Half Float in Integer)">;
def NoStdExtZhinx : Predicate<"!Subtarget->hasStdExtZhinx()">;

def HasStdExtZhinxOrZhinxmin
    : Predicate<"Subtarget->hasStdExtZhinxmin()">,
                AssemblerPredicate<(all_of FeatureStdExtZhinxmin),
                                   "'Zhinx' (Half Float in Integer) or "
                                   "'Zhinxmin' (Half Float in Integer Minimal)">;

def FeatureStdExtZfa
    : SubtargetFeature<"zfa", "HasStdExtZfa", "true",
                       "'Zfa' (Additional Floating-Point)",
                       [FeatureStdExtF]>;
def HasStdExtZfa : Predicate<"Subtarget->hasStdExtZfa()">,
                             AssemblerPredicate<(all_of FeatureStdExtZfa),
                             "'Zfa' (Additional Floating-Point)">;

def FeatureStdExtC
    : SubtargetFeature<"c", "HasStdExtC", "true",
                       "'C' (Compressed Instructions)">;
def HasStdExtC : Predicate<"Subtarget->hasStdExtC()">,
                           AssemblerPredicate<(all_of FeatureStdExtC),
                           "'C' (Compressed Instructions)">;

def FeatureStdExtZba
    : SubtargetFeature<"zba", "HasStdExtZba", "true",
                       "'Zba' (Address Generation Instructions)">;
def HasStdExtZba : Predicate<"Subtarget->hasStdExtZba()">,
                             AssemblerPredicate<(all_of FeatureStdExtZba),
                             "'Zba' (Address Generation Instructions)">;
def NotHasStdExtZba : Predicate<"!Subtarget->hasStdExtZba()">;

def FeatureStdExtZbb
    : SubtargetFeature<"zbb", "HasStdExtZbb", "true",
                       "'Zbb' (Basic Bit-Manipulation)">;
def HasStdExtZbb : Predicate<"Subtarget->hasStdExtZbb()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbb),
                             "'Zbb' (Basic Bit-Manipulation)">;

def FeatureStdExtZbc
    : SubtargetFeature<"zbc", "HasStdExtZbc", "true",
                       "'Zbc' (Carry-Less Multiplication)">;
def HasStdExtZbc : Predicate<"Subtarget->hasStdExtZbc()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbc),
                             "'Zbc' (Carry-Less Multiplication)">;

def FeatureStdExtZbs
    : SubtargetFeature<"zbs", "HasStdExtZbs", "true",
                       "'Zbs' (Single-Bit Instructions)">;
def HasStdExtZbs : Predicate<"Subtarget->hasStdExtZbs()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbs),
                             "'Zbs' (Single-Bit Instructions)">;

def FeatureStdExtZbkb
    : SubtargetFeature<"zbkb", "HasStdExtZbkb", "true",
                       "'Zbkb' (Bitmanip instructions for Cryptography)">;
def HasStdExtZbkb : Predicate<"Subtarget->hasStdExtZbkb()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbkb),
                             "'Zbkb' (Bitmanip instructions for Cryptography)">;

def FeatureStdExtZbkx
    : SubtargetFeature<"zbkx", "HasStdExtZbkx", "true",
                       "'Zbkx' (Crossbar permutation instructions)">;
def HasStdExtZbkx : Predicate<"Subtarget->hasStdExtZbkx()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbkx),
                             "'Zbkx' (Crossbar permutation instructions)">;

def HasStdExtZbbOrZbkb
    : Predicate<"Subtarget->hasStdExtZbb() || Subtarget->hasStdExtZbkb()">,
                AssemblerPredicate<(any_of FeatureStdExtZbb, FeatureStdExtZbkb),
                                   "'Zbb' (Basic Bit-Manipulation) or "
                                   "'Zbkb' (Bitmanip instructions for Cryptography)">;

// The Carry-less multiply subextension for cryptography is a subset of basic
// carry-less multiply subextension. The former should be enabled if the latter
// is enabled.
def FeatureStdExtZbkc
    : SubtargetFeature<"zbkc", "HasStdExtZbkc", "true",
                       "'Zbkc' (Carry-less multiply instructions for "
                       "Cryptography)">;
def HasStdExtZbkc
    : Predicate<"Subtarget->hasStdExtZbkc()">,
                AssemblerPredicate<(all_of FeatureStdExtZbkc),
                "'Zbkc' (Carry-less multiply instructions for Cryptography)">;

def HasStdExtZbcOrZbkc
    : Predicate<"Subtarget->hasStdExtZbc() || Subtarget->hasStdExtZbkc()">,
                AssemblerPredicate<(any_of FeatureStdExtZbc, FeatureStdExtZbkc),
                                   "'Zbc' (Carry-Less Multiplication) or "
                                   "'Zbkc' (Carry-less multiply instructions "
                                   "for Cryptography)">;

def FeatureStdExtZknd
    : SubtargetFeature<"zknd", "HasStdExtZknd", "true",
                       "'Zknd' (NIST Suite: AES Decryption)">;
def HasStdExtZknd : Predicate<"Subtarget->hasStdExtZknd()">,
                              AssemblerPredicate<(all_of FeatureStdExtZknd),
                              "'Zknd' (NIST Suite: AES Decryption)">;

def FeatureStdExtZkne
    : SubtargetFeature<"zkne", "HasStdExtZkne", "true",
                       "'Zkne' (NIST Suite: AES Encryption)">;
def HasStdExtZkne : Predicate<"Subtarget->hasStdExtZkne()">,
                              AssemblerPredicate<(all_of FeatureStdExtZkne),
                              "'Zkne' (NIST Suite: AES Encryption)">;

// Some instructions belong to both Zknd and Zkne subextensions.
// They should be enabled if either has been specified.
def HasStdExtZkndOrZkne
    : Predicate<"Subtarget->hasStdExtZknd() || Subtarget->hasStdExtZkne()">,
                AssemblerPredicate<(any_of FeatureStdExtZknd, FeatureStdExtZkne),
                                   "'Zknd' (NIST Suite: AES Decryption) or "
                                   "'Zkne' (NIST Suite: AES Encryption)">;

def FeatureStdExtZknh
    : SubtargetFeature<"zknh", "HasStdExtZknh", "true",
                       "'Zknh' (NIST Suite: Hash Function Instructions)">;
def HasStdExtZknh : Predicate<"Subtarget->hasStdExtZknh()">,
                             AssemblerPredicate<(all_of FeatureStdExtZknh),
                             "'Zknh' (NIST Suite: Hash Function Instructions)">;

def FeatureStdExtZksed
    : SubtargetFeature<"zksed", "HasStdExtZksed", "true",
                       "'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions)">;
def HasStdExtZksed : Predicate<"Subtarget->hasStdExtZksed()">,
                             AssemblerPredicate<(all_of FeatureStdExtZksed),
                             "'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions)">;

def FeatureStdExtZksh
    : SubtargetFeature<"zksh", "HasStdExtZksh", "true",
                       "'Zksh' (ShangMi Suite: SM3 Hash Function Instructions)">;
def HasStdExtZksh : Predicate<"Subtarget->hasStdExtZksh()">,
                              AssemblerPredicate<(all_of FeatureStdExtZksh),
                              "'Zksh' (ShangMi Suite: SM3 Hash Function "
                              "Instructions)">;

def FeatureStdExtZkr
    : SubtargetFeature<"zkr", "HasStdExtZkr", "true",
                       "'Zkr' (Entropy Source Extension)">;
def HasStdExtZkr : Predicate<"Subtarget->hasStdExtZkr()">,
                             AssemblerPredicate<(all_of FeatureStdExtZkr),
                             "'Zkr' (Entropy Source Extension)">;

def FeatureStdExtZkn
    : SubtargetFeature<"zkn", "HasStdExtZkn", "true",
                       "'Zkn' (NIST Algorithm Suite)",
                       [FeatureStdExtZbkb,
                        FeatureStdExtZbkc,
                        FeatureStdExtZbkx,
                        FeatureStdExtZkne,
                        FeatureStdExtZknd,
                        FeatureStdExtZknh]>;

def FeatureStdExtZks
    : SubtargetFeature<"zks", "HasStdExtZks", "true",
                       "'Zks' (ShangMi Algorithm Suite)",
                       [FeatureStdExtZbkb,
                        FeatureStdExtZbkc,
                        FeatureStdExtZbkx,
                        FeatureStdExtZksed,
                        FeatureStdExtZksh]>;

def FeatureStdExtZkt
    : SubtargetFeature<"zkt", "HasStdExtZkt", "true",
                       "'Zkt' (Data Independent Execution Latency)">;

def FeatureStdExtZk
    : SubtargetFeature<"zk", "HasStdExtZk", "true",
                       "'Zk' (Standard scalar cryptography extension)",
                       [FeatureStdExtZkn,
                        FeatureStdExtZkr,
                        FeatureStdExtZkt]>;

def FeatureStdExtZca
    : SubtargetFeature<"zca", "HasStdExtZca", "true",
                       "'Zca' (part of the C extension, excluding compressed "
                       "floating point loads/stores)">;

def HasStdExtCOrZca
    : Predicate<"Subtarget->hasStdExtCOrZca()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZca),
                                   "'C' (Compressed Instructions) or "
                                   "'Zca' (part of the C extension, excluding "
                                   "compressed floating point loads/stores)">;

def FeatureStdExtZcb
    : SubtargetFeature<"zcb", "HasStdExtZcb", "true",
                       "'Zcb' (Compressed basic bit manipulation instructions)",
                       [FeatureStdExtZca]>;
def HasStdExtZcb : Predicate<"Subtarget->hasStdExtZcb()">,
                             AssemblerPredicate<(all_of FeatureStdExtZcb),
                             "'Zcb' (Compressed basic bit manipulation instructions)">;

def FeatureStdExtZcd
    : SubtargetFeature<"zcd", "HasStdExtZcd", "true",
                       "'Zcd' (Compressed Double-Precision Floating-Point Instructions)",
                       [FeatureStdExtZca]>;

def HasStdExtCOrZcd
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcd()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZcd),
                                   "'C' (Compressed Instructions) or "
                                   "'Zcd' (Compressed Double-Precision Floating-Point Instructions)">;

def FeatureStdExtZcf
    : SubtargetFeature<"zcf", "HasStdExtZcf", "true",
                       "'Zcf' (Compressed Single-Precision Floating-Point Instructions)",
                       [FeatureStdExtZca]>;

def FeatureStdExtZcmp
    : SubtargetFeature<"zcmp", "HasStdExtZcmp", "true",
                       "'Zcmp' (sequenced instuctions for code-size reduction)",
                       [FeatureStdExtZca]>;
def HasStdExtZcmp : Predicate<"Subtarget->hasStdExtZcmp() && !Subtarget->hasStdExtC()">,
                               AssemblerPredicate<(all_of FeatureStdExtZcmp),
                               "'Zcmp' (sequenced instuctions for code-size reduction)">;

def FeatureStdExtZcmt
    : SubtargetFeature<"zcmt", "HasStdExtZcmt", "true",
                       "'Zcmt' (table jump instuctions for code-size reduction)",
                       [FeatureStdExtZca, FeatureStdExtZicsr]>;
def HasStdExtZcmt : Predicate<"Subtarget->hasStdExtZcmt()">,
                           AssemblerPredicate<(all_of FeatureStdExtZcmt),
                           "'Zcmt' (table jump instuctions for code-size reduction)">;

def FeatureStdExtZce
    : SubtargetFeature<"zce", "HasStdExtZce", "true",
                       "'Zce' (Compressed extensions for microcontrollers)",
                       [FeatureStdExtZca, FeatureStdExtZcb, FeatureStdExtZcmp,
                        FeatureStdExtZcmt]>;

def HasStdExtCOrZcfOrZce
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcf() "
                "Subtarget->hasStdExtZce()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZcf,
                                           FeatureStdExtZce),
                                   "'C' (Compressed Instructions) or "
                                   "'Zcf' (Compressed Single-Precision Floating-Point Instructions)">;

def FeatureNoRVCHints
    : SubtargetFeature<"no-rvc-hints", "EnableRVCHintInstrs", "false",
                       "Disable RVC Hint Instructions.">;
def HasRVCHints : Predicate<"Subtarget->enableRVCHintInstrs()">,
                  AssemblerPredicate<(all_of(not FeatureNoRVCHints)),
                                      "RVC Hint Instructions">;

def FeatureStdExtZvl32b : SubtargetFeature<"zvl32b", "ZvlLen", "32",
                                           "'Zvl' (Minimum Vector Length) 32">;

foreach i = { 6-16 } in {
  defvar I = !shl(1, i);
  def FeatureStdExtZvl#I#b :
      SubtargetFeature<"zvl"#I#"b", "ZvlLen", !cast<string>(I),
                       "'Zvl' (Minimum Vector Length) "#I,
                       [!cast<SubtargetFeature>("FeatureStdExtZvl"#!srl(I, 1)#"b")]>;
}

def FeatureStdExtZve32x
    : SubtargetFeature<"zve32x", "HasStdExtZve32x", "true",
                       "'Zve32x' (Vector Extensions for Embedded Processors "
                       "with maximal 32 EEW)",
                       [FeatureStdExtZicsr, FeatureStdExtZvl32b]>;

def FeatureStdExtZve32f
    : SubtargetFeature<"zve32f", "HasStdExtZve32f", "true",
                       "'Zve32f' (Vector Extensions for Embedded Processors "
                       "with maximal 32 EEW and F extension)",
                       [FeatureStdExtZve32x, FeatureStdExtF]>;

def FeatureStdExtZve64x
    : SubtargetFeature<"zve64x", "HasStdExtZve64x", "true",
                       "'Zve64x' (Vector Extensions for Embedded Processors "
                       "with maximal 64 EEW)",
                       [FeatureStdExtZve32x, FeatureStdExtZvl64b]>;

def FeatureStdExtZve64f
    : SubtargetFeature<"zve64f", "HasStdExtZve64f", "true",
                       "'Zve64f' (Vector Extensions for Embedded Processors "
                       "with maximal 64 EEW and F extension)",
                       [FeatureStdExtZve32f, FeatureStdExtZve64x]>;

def FeatureStdExtZve64d
    : SubtargetFeature<"zve64d", "HasStdExtZve64d", "true",
                       "'Zve64d' (Vector Extensions for Embedded Processors "
                       "with maximal 64 EEW, F and D extension)",
                       [FeatureStdExtZve64f, FeatureStdExtD]>;

def FeatureStdExtV
    : SubtargetFeature<"v", "HasStdExtV", "true",
                       "'V' (Vector Extension for Application Processors)",
                       [FeatureStdExtZvl128b, FeatureStdExtZve64d]>;

def HasVInstructions    : Predicate<"Subtarget->hasVInstructions()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve32x),
          "'V' (Vector Extension for Application Processors), 'Zve32x' "
          "(Vector Extensions for Embedded Processors)">;
def HasVInstructionsI64 : Predicate<"Subtarget->hasVInstructionsI64()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve64x),
          "'V' (Vector Extension for Application Processors) or 'Zve64x' "
          "(Vector Extensions for Embedded Processors)">;
def HasVInstructionsAnyF : Predicate<"Subtarget->hasVInstructionsAnyF()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve32f),
          "'V' (Vector Extension for Application Processors), 'Zve32f' "
          "(Vector Extensions for Embedded Processors)">;

def HasVInstructionsF64 : Predicate<"Subtarget->hasVInstructionsF64()">;

def HasVInstructionsFullMultiply : Predicate<"Subtarget->hasVInstructionsFullMultiply()">;

def FeatureStdExtZfbfmin
    : SubtargetFeature<"experimental-zfbfmin", "HasStdExtZfbfmin", "true",
                       "'Zfbfmin' (Scalar BF16 Converts)",
                       [FeatureStdExtF]>;
def HasStdExtZfbfmin : Predicate<"Subtarget->hasStdExtZfbfmin()">,
                                 AssemblerPredicate<(all_of FeatureStdExtZfbfmin),
                                 "'Zfbfmin' (Scalar BF16 Converts)">;

def FeatureStdExtZvfbfmin
    : SubtargetFeature<"experimental-zvfbfmin", "HasStdExtZvfbfmin", "true",
                       "'Zvbfmin' (Vector BF16 Converts)",
                       [FeatureStdExtZve32f]>;
def HasStdExtZvfbfmin : Predicate<"Subtarget->hasStdExtZvfbfmin()">,
                                  AssemblerPredicate<(all_of FeatureStdExtZvfbfmin),
                                  "'Zvfbfmin' (Vector BF16 Converts)">;

def FeatureStdExtZvfbfwma
    : SubtargetFeature<"experimental-zvfbfwma", "HasStdExtZvfbfwma", "true",
                       "'Zvfbfwma' (Vector BF16 widening mul-add)",
                       [FeatureStdExtZvfbfmin, FeatureStdExtZfbfmin]>;
def HasStdExtZvfbfwma : Predicate<"Subtarget->hasStdExtZvfbfwma()">,
                                  AssemblerPredicate<(all_of FeatureStdExtZvfbfwma),
                                  "'Zvfbfwma' (Vector BF16 widening mul-add)">;

def HasVInstructionsBF16 : Predicate<"Subtarget->hasVInstructionsBF16()">;

def FeatureStdExtZvfhmin
    : SubtargetFeature<"zvfhmin", "HasStdExtZvfhmin", "true",
                       "'Zvfhmin' (Vector Half-Precision Floating-Point Minimal)",
                       [FeatureStdExtZve32f]>;

def FeatureStdExtZvfh
    : SubtargetFeature<"zvfh", "HasStdExtZvfh", "true",
                       "'Zvfh' (Vector Half-Precision Floating-Point)",
                       [FeatureStdExtZvfhmin, FeatureStdExtZfhmin]>;

def HasVInstructionsF16 : Predicate<"Subtarget->hasVInstructionsF16()">;

def HasVInstructionsF16Minimal : Predicate<"Subtarget->hasVInstructionsF16Minimal()">,
      AssemblerPredicate<(any_of FeatureStdExtZvfhmin, FeatureStdExtZvfh),
                         "'Zvfhmin' (Vector Half-Precision Floating-Point Minimal) or "
                         "'Zvfh' (Vector Half-Precision Floating-Point)">;

def HasStdExtZfhOrZvfh
    : Predicate<"Subtarget->hasStdExtZfh() || Subtarget->hasStdExtZvfh()">,
                AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZvfh),
                                   "'Zfh' (Half-Precision Floating-Point) or "
                                   "'Zvfh' (Vector Half-Precision Floating-Point)">;

def FeatureStdExtZicbom
    : SubtargetFeature<"zicbom", "HasStdExtZicbom", "true",
                       "'Zicbom' (Cache-Block Management Instructions)">;
def HasStdExtZicbom : Predicate<"Subtarget->hasStdExtZicbom()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicbom),
                                "'Zicbom' (Cache-Block Management Instructions)">;

def FeatureStdExtZicboz
    : SubtargetFeature<"zicboz", "HasStdExtZicboz", "true",
                       "'Zicboz' (Cache-Block Zero Instructions)">;
def HasStdExtZicboz : Predicate<"Subtarget->hasStdExtZicboz()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicboz),
                                "'Zicboz' (Cache-Block Zero Instructions)">;

def FeatureStdExtZicbop
    : SubtargetFeature<"zicbop", "HasStdExtZicbop", "true",
                       "'Zicbop' (Cache-Block Prefetch Instructions)">;
def HasStdExtZicbop : Predicate<"Subtarget->hasStdExtZicbop()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicbop),
                                "'Zicbop' (Cache-Block Prefetch Instructions)">;

def FeatureStdExtSvnapot
    : SubtargetFeature<"svnapot", "HasStdExtSvnapot", "true",
                       "'Svnapot' (NAPOT Translation Contiguity)">;

def FeatureStdExtSvpbmt
    : SubtargetFeature<"svpbmt", "HasStdExtSvpbmt", "true",
                       "'Svpbmt' (Page-Based Memory Types)">;

def FeatureStdExtSvinval
    : SubtargetFeature<"svinval", "HasStdExtSvinval", "true",
                       "'Svinval' (Fine-Grained Address-Translation Cache Invalidation)">;
def HasStdExtSvinval : Predicate<"Subtarget->hasStdExtSvinval()">,
                                AssemblerPredicate<(all_of FeatureStdExtSvinval),
                                "'Svinval' (Fine-Grained Address-Translation Cache Invalidation)">;

def FeatureStdExtZtso
    : SubtargetFeature<"experimental-ztso", "HasStdExtZtso", "true",
                       "'Ztso' (Memory Model - Total Store Order)">;
def HasStdExtZtso : Predicate<"Subtarget->hasStdExtZtso()">,
                              AssemblerPredicate<(all_of FeatureStdExtZtso),
                              "'Ztso' (Memory Model - Total Store Order)">;
def NotHasStdExtZtso : Predicate<"!Subtarget->hasStdExtZtso()">;

def FeatureStdExtZawrs : SubtargetFeature<"zawrs", "HasStdExtZawrs", "true",
                                          "'Zawrs' (Wait on Reservation Set)">;
def HasStdExtZawrs : Predicate<"Subtarget->hasStdExtZawrs()">,
                               AssemblerPredicate<(all_of FeatureStdExtZawrs),
                               "'Zawrs' (Wait on Reservation Set)">;

def FeatureStdExtZvkb
    : SubtargetFeature<"zvkb", "HasStdExtZvkb", "true",
                       "'Zvkb' (Vector Bit-manipulation used in Cryptography)">;
def HasStdExtZvkb : Predicate<"Subtarget->hasStdExtZvkb()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvkb),
                              "'Zvkb' (Vector Bit-manipulation used in Cryptography)">;

def FeatureStdExtZvbb
    : SubtargetFeature<"zvbb", "HasStdExtZvbb", "true",
                       "'Zvbb' (Vector basic bit-manipulation instructions.)",
                       [FeatureStdExtZvkb]>;
def HasStdExtZvbb : Predicate<"Subtarget->hasStdExtZvbb()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvbb),
                              "'Zvbb' (Vector basic bit-manipulation instructions.)">;

def FeatureStdExtZvbc
    : SubtargetFeature<"zvbc", "HasStdExtZvbc", "true",
                       "'Zvbc' (Vector Carryless Multiplication)">;
def HasStdExtZvbc : Predicate<"Subtarget->hasStdExtZvbc()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvbc),
                              "'Zvbc' (Vector Carryless Multiplication)">;

def FeatureStdExtZvkg
    : SubtargetFeature<"zvkg", "HasStdExtZvkg", "true",
                       "'Zvkg' (Vector GCM instructions for Cryptography)">;
def HasStdExtZvkg : Predicate<"Subtarget->hasStdExtZvkg()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvkg),
                              "'Zvkg' (Vector GCM instructions for Cryptography)">;

def FeatureStdExtZvkned
    : SubtargetFeature<"zvkned", "HasStdExtZvkned", "true",
                       "'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;
def HasStdExtZvkned : Predicate<"Subtarget->hasStdExtZvkned()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvkned),
                                "'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;

def FeatureStdExtZvknha
    : SubtargetFeature<"zvknha", "HasStdExtZvknha", "true",
                       "'Zvknha' (Vector SHA-2 (SHA-256 only))">;
def HasStdExtZvknha : Predicate<"Subtarget->hasStdExtZvknha()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvknha),
                                "'Zvknha' (Vector SHA-2 (SHA-256 only))">;

def FeatureStdExtZvknhb
    : SubtargetFeature<"zvknhb", "HasStdExtZvknhb", "true",
                       "'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512))",
                       [FeatureStdExtZve64x]>;
def HasStdExtZvknhb : Predicate<"Subtarget->hasStdExtZvknhb()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvknhb),
                                "'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512))">;

def HasStdExtZvknhaOrZvknhb : Predicate<"Subtarget->hasStdExtZvknha() || Subtarget->hasStdExtZvknhb()">,
                                        AssemblerPredicate<(any_of FeatureStdExtZvknha, FeatureStdExtZvknhb),
                                        "'Zvknha' or 'Zvknhb' (Vector SHA-2)">;

def FeatureStdExtZvksed
    : SubtargetFeature<"zvksed", "HasStdExtZvksed", "true",
                       "'Zvksed' (SM4 Block Cipher Instructions)">;
def HasStdExtZvksed : Predicate<"Subtarget->hasStdExtZvksed()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvksed),
                                "'Zvksed' (SM4 Block Cipher Instructions)">;

def FeatureStdExtZvksh
    : SubtargetFeature<"zvksh", "HasStdExtZvksh", "true",
                       "'Zvksh' (SM3 Hash Function Instructions)">;
def HasStdExtZvksh : Predicate<"Subtarget->hasStdExtZvksh()">,
                               AssemblerPredicate<(all_of FeatureStdExtZvksh),
                               "'Zvksh' (SM3 Hash Function Instructions)">;

def FeatureStdExtZvkt
    : SubtargetFeature<"zvkt", "HasStdExtZvkt", "true",
                       "'Zvkt' (Vector Data-Independent Execution Latency)">;

// Zvk short-hand extensions

def FeatureStdExtZvkn
    : SubtargetFeature<"zvkn", "HasStdExtZvkn", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvkned, Zvknhb, Zvkb and Zvkt.",
                       [FeatureStdExtZvkned, FeatureStdExtZvknhb,
                        FeatureStdExtZvkb, FeatureStdExtZvkt]>;

def FeatureStdExtZvknc
    : SubtargetFeature<"zvknc", "HasStdExtZvknc", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvkn and Zvbc.",
                       [FeatureStdExtZvkn, FeatureStdExtZvbc]>;

def FeatureStdExtZvkng
    : SubtargetFeature<"zvkng", "HasStdExtZvkng", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvkn and Zvkg.",
                       [FeatureStdExtZvkn, FeatureStdExtZvkg]>;

def FeatureStdExtZvks
    : SubtargetFeature<"zvks", "HasStdExtZvks", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvksed, Zvksh, Zvkb and Zvkt.",
                       [FeatureStdExtZvksed, FeatureStdExtZvksh,
                        FeatureStdExtZvkb, FeatureStdExtZvkt]>;

def FeatureStdExtZvksc
    : SubtargetFeature<"zvksc", "HasStdExtZvksc", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvks and Zvbc.",
                       [FeatureStdExtZvks, FeatureStdExtZvbc]>;

def FeatureStdExtZvksg
    : SubtargetFeature<"zvksg", "HasStdExtZvksg", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvks and Zvkg.",
                       [FeatureStdExtZvks, FeatureStdExtZvkg]>;

def FeatureStdExtZicfilp
    : SubtargetFeature<"experimental-zicfilp", "HasStdExtZicfilp", "true",
                       "'Zicfilp' (Landing pad)">;
def HasStdExtZicfilp : Predicate<"Subtarget->hasStdExtZicfilp()">,
                                 AssemblerPredicate<(all_of FeatureStdExtZicfilp),
                                 "'Zicfilp' (Landing pad)">;

def FeatureStdExtZicond
    : SubtargetFeature<"experimental-zicond", "HasStdExtZicond", "true",
                       "'Zicond' (Integer Conditional Operations)">;
def HasStdExtZicond : Predicate<"Subtarget->hasStdExtZicond()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicond),
                                "'Zicond' (Integer Conditional Operations)">;

def FeatureStdExtZimop : SubtargetFeature<"experimental-zimop", "HasStdExtZimop", "true",
                                          "'Zimop' (May-Be-Operations)">;
def HasStdExtZimop : Predicate<"Subtarget->hasStdExtZimop()">,
                               AssemblerPredicate<(all_of FeatureStdExtZimop),
                               "'Zimop' (May-Be-Operations)">;

def FeatureStdExtZcmop : SubtargetFeature<"experimental-zcmop", "HasStdExtZcmop", "true",
                                          "'Zcmop' (Compressed May-Be-Operations)",
                                          [FeatureStdExtZca]>;
def HasStdExtZcmop : Predicate<"Subtarget->hasStdExtZcmop()">,
                               AssemblerPredicate<(all_of FeatureStdExtZcmop),
                               "'Zcmop' (Compressed May-Be-Operations)">;

def FeatureStdExtZicfiss
    : SubtargetFeature<"experimental-zicfiss", "HasStdExtZicfiss", "true",
                       "'Zicfiss' (Shadow stack)",
                       [FeatureStdExtZicsr, FeatureStdExtZimop]>;
def HasStdExtZicfiss : Predicate<"Subtarget->hasStdExtZicfiss()">,
                                 AssemblerPredicate<(all_of FeatureStdExtZicfiss),
                                 "'Zicfiss' (Shadow stack)">;
def NoHasStdExtZicfiss : Predicate<"!Subtarget->hasStdExtZicfiss()">;

def FeatureStdExtSmaia
    : SubtargetFeature<"smaia", "HasStdExtSmaia", "true",
                       "'Smaia' (Smaia encompasses all added CSRs and all "
                       "modifications to interrupt response behavior that the "
                       "AIA specifies for a hart, over all privilege levels.)",
                       []>;

def FeatureStdExtSsaia
    : SubtargetFeature<"ssaia", "HasStdExtSsaia", "true",
                       "'Ssaia' (Ssaia is essentially the same as Smaia except "
                       "excluding the machine-level CSRs and behavior not "
                       "directly visible to supervisor level.)", []>;

def HasHalfFPLoadStoreMove
    : Predicate<"Subtarget->hasHalfFPLoadStoreMove()">,
                AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZfhmin,
                                    FeatureStdExtZfbfmin),
                                    "'Zfh' (Half-Precision Floating-Point) or "
                                    "'Zfhmin' (Half-Precision Floating-Point Minimal) or "
                                    "'Zfbfmin' (Scalar BF16 Converts)">;

def FeatureStdExtZacas
    : SubtargetFeature<"experimental-zacas", "HasStdExtZacas", "true",
                       "'Zacas' (Atomic Compare-And-Swap Instructions)">;
def HasStdExtZacas : Predicate<"Subtarget->hasStdExtZacas()">,
                               AssemblerPredicate<(all_of FeatureStdExtZacas),
                               "'Zacas' (Atomic Compare-And-Swap Instructions)">;

//===----------------------------------------------------------------------===//
// Vendor extensions
//===----------------------------------------------------------------------===//

def FeatureVendorXVentanaCondOps
    : SubtargetFeature<"xventanacondops", "HasVendorXVentanaCondOps", "true",
                       "'XVentanaCondOps' (Ventana Conditional Ops)">;
def HasVendorXVentanaCondOps : Predicate<"Subtarget->hasVendorXVentanaCondOps()">,
                                AssemblerPredicate<(all_of FeatureVendorXVentanaCondOps),
                                "'XVentanaCondOps' (Ventana Conditional Ops)">;

def FeatureVendorXTHeadBa
    : SubtargetFeature<"xtheadba", "HasVendorXTHeadBa", "true",
                       "'xtheadba' (T-Head address calculation instructions)">;
def HasVendorXTHeadBa : Predicate<"Subtarget->hasVendorXTHeadBa()">,
                                  AssemblerPredicate<(all_of FeatureVendorXTHeadBa),
                                  "'xtheadba' (T-Head address calculation instructions)">;

def FeatureVendorXTHeadBb
    : SubtargetFeature<"xtheadbb", "HasVendorXTHeadBb", "true",
                       "'xtheadbb' (T-Head basic bit-manipulation instructions)">;
def HasVendorXTHeadBb : Predicate<"Subtarget->hasVendorXTHeadBb()">,
                                  AssemblerPredicate<(all_of FeatureVendorXTHeadBb),
                                  "'xtheadbb' (T-Head basic bit-manipulation instructions)">;

def FeatureVendorXTHeadBs
    : SubtargetFeature<"xtheadbs", "HasVendorXTHeadBs", "true",
                       "'xtheadbs' (T-Head single-bit instructions)">;
def HasVendorXTHeadBs : Predicate<"Subtarget->hasVendorXTHeadBs()">,
                                  AssemblerPredicate<(all_of FeatureVendorXTHeadBs),
                                  "'xtheadbs' (T-Head single-bit instructions)">;

def FeatureVendorXTHeadCondMov
    : SubtargetFeature<"xtheadcondmov", "HasVendorXTHeadCondMov", "true",
                       "'xtheadcondmov' (T-Head conditional move instructions)">;
def HasVendorXTHeadCondMov : Predicate<"Subtarget->hasVendorXTHeadCondMov()">,
                                       AssemblerPredicate<(all_of FeatureVendorXTHeadCondMov),
                                       "'xtheadcondmov' (T-Head conditional move instructions)">;

def FeatureVendorXTHeadCmo
    : SubtargetFeature<"xtheadcmo", "HasVendorXTHeadCmo", "true",
                       "'xtheadcmo' (T-Head cache management instructions)">;
def HasVendorXTHeadCmo : Predicate<"Subtarget->hasVendorXTHeadCmo()">,
                                   AssemblerPredicate<(all_of FeatureVendorXTHeadCmo),
                                   "'xtheadcmo' (T-Head cache management instructions)">;

def FeatureVendorXTHeadFMemIdx
    : SubtargetFeature<"xtheadfmemidx", "HasVendorXTHeadFMemIdx", "true",
                       "'xtheadfmemidx' (T-Head FP Indexed Memory Operations)",
                       [FeatureStdExtF]>;
def HasVendorXTHeadFMemIdx : Predicate<"Subtarget->hasVendorXTHeadFMemIdx()">,
                                       AssemblerPredicate<(all_of FeatureVendorXTHeadFMemIdx),
                                       "'xtheadfmemidx' (T-Head FP Indexed Memory Operations)">;

def FeatureVendorXTHeadMac
    : SubtargetFeature<"xtheadmac", "HasVendorXTHeadMac", "true",
                       "'xtheadmac' (T-Head Multiply-Accumulate Instructions)">;
def HasVendorXTHeadMac : Predicate<"Subtarget->hasVendorXTHeadMac()">,
                                   AssemblerPredicate<(all_of FeatureVendorXTHeadMac),
                                   "'xtheadmac' (T-Head Multiply-Accumulate Instructions)">;

def FeatureVendorXTHeadMemIdx
    : SubtargetFeature<"xtheadmemidx", "HasVendorXTHeadMemIdx", "true",
                       "'xtheadmemidx' (T-Head Indexed Memory Operations)">;
def HasVendorXTHeadMemIdx : Predicate<"Subtarget->hasVendorXTHeadMemIdx()">,
                                      AssemblerPredicate<(all_of FeatureVendorXTHeadMemIdx),
                                      "'xtheadmemidx' (T-Head Indexed Memory Operations)">;

def FeatureVendorXTHeadMemPair
    : SubtargetFeature<"xtheadmempair", "HasVendorXTHeadMemPair", "true",
                       "'xtheadmempair' (T-Head two-GPR Memory Operations)">;
def HasVendorXTHeadMemPair : Predicate<"Subtarget->hasVendorXTHeadMemPair()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadMemPair),
                                    "'xtheadmempair' (T-Head two-GPR Memory Operations)">;

def FeatureVendorXTHeadSync
    : SubtargetFeature<"xtheadsync", "HasVendorXTHeadSync", "true",
                       "'xtheadsync' (T-Head multicore synchronization instructions)">;
def HasVendorXTHeadSync : Predicate<"Subtarget->hasVendorXTHeadSync()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadSync),
                                    "'xtheadsync' (T-Head multicore synchronization instructions)">;

def FeatureVendorXTHeadVdot
    : SubtargetFeature<"xtheadvdot", "HasVendorXTHeadVdot", "true",
                       "'xtheadvdot' (T-Head Vector Extensions for Dot)",
                       [FeatureStdExtV]>;
def HasVendorXTHeadVdot : Predicate<"Subtarget->hasVendorXTHeadVdot()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadVdot),
                                    "'xtheadvdot' (T-Head Vector Extensions for Dot)">;

def FeatureVendorXSfvcp
    : SubtargetFeature<"xsfvcp", "HasVendorXSfvcp", "true",
                       "'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions)",
                       [FeatureStdExtZve32x]>;
def HasVendorXSfvcp : Predicate<"Subtarget->hasVendorXSfvcp()">,
                                AssemblerPredicate<(all_of FeatureVendorXSfvcp),
                                "'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions)">;

def FeatureVendorXSfvqmaccdod
    : SubtargetFeature<"xsfvqmaccdod", "HasVendorXSfvqmaccdod", "true",
                       "'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))",
                       [FeatureStdExtZve32x]>;
def HasVendorXSfvqmaccdod : Predicate<"Subtarget->hasVendorXSfvqmaccdod()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvqmaccdod),
                         "'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))">;

def FeatureVendorXSfvqmaccqoq
    : SubtargetFeature<"xsfvqmaccqoq", "HasVendorXSfvqmaccqoq", "true",
                       "'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))",
                       [FeatureStdExtZve32x]>;
def HasVendorXSfvqmaccqoq : Predicate<"Subtarget->hasVendorXSfvqmaccqoq()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvqmaccqoq),
                         "'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))">;

def FeatureVendorXSfvfwmaccqqq
    : SubtargetFeature<"xsfvfwmaccqqq", "HasVendorXSfvfwmaccqqq", "true",
                       "'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction and 4-by-4))",
                       [FeatureStdExtZve32f, FeatureStdExtZvfbfmin]>;
def HasVendorXSfvfwmaccqqq : Predicate<"Subtarget->hasVendorXSfvfwmaccqqq()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvfwmaccqqq),
                         "'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction and 4-by-4))">;

def FeatureVendorXSfvfnrclipxfqf
    : SubtargetFeature<"xsfvfnrclipxfqf", "HasVendorXSfvfnrclipxfqf", "true",
                       "'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)",
                       [FeatureStdExtZve32f]>;
def HasVendorXSfvfnrclipxfqf : Predicate<"Subtarget->hasVendorXSfvfnrclipxfqf()">,
                               AssemblerPredicate<(all_of FeatureVendorXSfvfnrclipxfqf),
                               "'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)">;
def FeatureVendorXCVelw
   : SubtargetFeature<"xcvelw", "HasVendorXCVelw", "true",
                      "'XCVelw' (CORE-V Event Load Word)">;
def HasVendorXCVelw
   : Predicate<"Subtarget->hasVendorXCVelw()">,
              AssemblerPredicate<(any_of FeatureVendorXCVelw),
              "'XCVelw' (CORE-V Event Load Word)">;

def FeatureVendorXCVbitmanip
    : SubtargetFeature<"xcvbitmanip", "HasVendorXCVbitmanip", "true",
                       "'XCVbitmanip' (CORE-V Bit Manipulation)">;
def HasVendorXCVbitmanip : Predicate<"Subtarget->hasVendorXCVbitmanip()">,
                                AssemblerPredicate<(all_of FeatureVendorXCVbitmanip),
                                "'XCVbitmanip' (CORE-V Bit Manipulation)">;

def FeatureVendorXCVmac
    : SubtargetFeature<"xcvmac", "HasVendorXCVmac", "true",
                       "'XCVmac' (CORE-V Multiply-Accumulate)">;
def HasVendorXCVmac : Predicate<"Subtarget->hasVendorXCVmac()">,
                                AssemblerPredicate<(all_of FeatureVendorXCVmac),
                                "'XCVmac' (CORE-V Multiply-Accumulate)">;

def FeatureVendorXCVmem
    : SubtargetFeature<"xcvmem", "HasVendorXCVmem", "true",
                       "'XCVmem' (CORE-V Post-incrementing Load & Store)">;
def HasVendorXCVmem
    : Predicate<"Subtarget->hasVendorXCVmem()">,
               AssemblerPredicate<(any_of FeatureVendorXCVmem),
               "'XCVmem' (CORE-V Post-incrementing Load & Store)">;

def FeatureVendorXCValu
    : SubtargetFeature<"xcvalu", "HasVendorXCValu", "true",
                       "'XCValu' (CORE-V ALU Operations)">;
def HasVendorXCValu : Predicate<"Subtarget->hasVendorXCValu()">,
                                AssemblerPredicate<(all_of FeatureVendorXCValu),
                                "'XCValu' (CORE-V ALU Operations)">;

def FeatureVendorXCVsimd
    : SubtargetFeature<"xcvsimd", "HasVendorXCvsimd", "true",
                       "'XCVsimd' (CORE-V SIMD ALU)">;
def HasVendorXCVsimd
    : Predicate<"Subtarget->hasVendorXCVsimd()">,
                AssemblerPredicate<(any_of FeatureVendorXCVsimd),
                "'XCVsimd' (CORE-V SIMD ALU)">;

def FeatureVendorXCVbi
    : SubtargetFeature<"xcvbi", "HasVendorXCVbi", "true",
                       "'XCVbi' (CORE-V Immediate Branching)">;
def HasVendorXCVbi : Predicate<"Subtarget->hasVendorXCVbi()">,
                               AssemblerPredicate<(all_of FeatureVendorXCVbi),
                               "'XCVbi' (CORE-V Immediate Branching)">;

//===----------------------------------------------------------------------===//
// LLVM specific features and extensions
//===----------------------------------------------------------------------===//

// Feature32Bit exists to mark CPUs that support RV32 to distinquish them from
// tuning CPU names.
def Feature32Bit
    : SubtargetFeature<"32bit", "IsRV32", "true", "Implements RV32">;
def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of Feature64Bit),
                                         "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of (not Feature64Bit)),
                                          "RV32I Base Instruction Set">;

defvar RV32 = DefaultMode;
def RV64           : HwMode<"+64bit", [IsRV64]>;

def FeatureRVE
    : SubtargetFeature<"e", "IsRVE", "true",
                       "Implements RV{32,64}E (provides 16 rather than 32 GPRs)">;
def IsRVE : Predicate<"Subtarget->isRVE()">,
                        AssemblerPredicate<(all_of FeatureRVE)>;

def FeatureRelax
    : SubtargetFeature<"relax", "EnableLinkerRelax", "true",
                       "Enable Linker relaxation.">;

foreach i = {1-31} in
  def FeatureReserveX#i :
      SubtargetFeature<"reserve-x"#i, "UserReservedRegister[RISCV::X"#i#"]",
                       "true", "Reserve X"#i>;

def FeatureSaveRestore : SubtargetFeature<"save-restore", "EnableSaveRestore",
                                          "true", "Enable save/restore.">;

def FeatureTrailingSeqCstFence : SubtargetFeature<"seq-cst-trailing-fence",
                                          "EnableSeqCstTrailingFence",
                                          "true",
                                          "Enable trailing fence for seq-cst store.">;

def FeatureFastUnalignedAccess
   : SubtargetFeature<"fast-unaligned-access", "HasFastUnalignedAccess",
                      "true", "Has reasonably performant unaligned "
                      "loads and stores (both scalar and vector)">;

def FeaturePostRAScheduler : SubtargetFeature<"use-postra-scheduler",
    "UsePostRAScheduler", "true", "Schedule again after register allocation">;

def TuneNoOptimizedZeroStrideLoad
   : SubtargetFeature<"no-optimized-zero-stride-load", "HasOptimizedZeroStrideLoad",
                      "false", "Hasn't optimized (perform fewer memory operations)"
                      "zero-stride vector load">;

def Experimental
   : SubtargetFeature<"experimental", "HasExperimental",
                      "true", "Experimental intrinsics">;

// Some vector hardware implementations do not process all VLEN bits in parallel
// and instead split over multiple cycles. DLEN refers to the datapath width
// that can be done in parallel.
def TuneDLenFactor2
   : SubtargetFeature<"dlen-factor-2", "DLenFactor2", "true",
                      "Vector unit DLEN(data path width) is half of VLEN">;

def TuneLUIADDIFusion
    : SubtargetFeature<"lui-addi-fusion", "HasLUIADDIFusion",
                       "true", "Enable LUI+ADDI macrofusion">;

def TuneAUIPCADDIFusion
    : SubtargetFeature<"auipc-addi-fusion", "HasAUIPCADDIFusion",
                       "true", "Enable AUIPC+ADDI macrofusion">;

def TuneZExtHFusion
    : SubtargetFeature<"zexth-fusion", "HasZExtHFusion",
                       "true", "Enable SLLI+SRLI to be fused to zero extension of halfword">;

def TuneZExtWFusion
    : SubtargetFeature<"zextw-fusion", "HasZExtWFusion",
                       "true", "Enable SLLI+SRLI to be fused to zero extension of word">;

def TuneShiftedZExtWFusion
    : SubtargetFeature<"shifted-zextw-fusion", "HasShiftedZExtWFusion",
                       "true", "Enable SLLI+SRLI to be fused when computing (shifted) zero extension of word">;

def TuneLDADDFusion
    : SubtargetFeature<"ld-add-fusion", "HasLDADDFusion",
                       "true", "Enable LD+ADD macrofusion.">;

def TuneNoDefaultUnroll
    : SubtargetFeature<"no-default-unroll", "EnableDefaultUnroll", "false",
                       "Disable default unroll preference.">;

// SiFive 7 is able to fuse integer ALU operations with a preceding branch
// instruction.
def TuneShortForwardBranchOpt
    : SubtargetFeature<"short-forward-branch-opt", "HasShortForwardBranchOpt",
                       "true", "Enable short forward branch optimization">;
def HasShortForwardBranchOpt : Predicate<"Subtarget->hasShortForwardBranchOpt()">;
def NoShortForwardBranchOpt : Predicate<"!Subtarget->hasShortForwardBranchOpt()">;

def TuneConditionalCompressedMoveFusion
    : SubtargetFeature<"conditional-cmv-fusion", "HasConditionalCompressedMoveFusion",
                       "true", "Enable branch+c.mv fusion">;
def HasConditionalMoveFusion : Predicate<"Subtarget->hasConditionalMoveFusion()">;
def NoConditionalMoveFusion  : Predicate<"!Subtarget->hasConditionalMoveFusion()">;

def TuneSiFive7 : SubtargetFeature<"sifive7", "RISCVProcFamily", "SiFive7",
                                   "SiFive 7-Series processors",
                                   [TuneNoDefaultUnroll,
                                    TuneShortForwardBranchOpt]>;

def TuneVentanaVeyron : SubtargetFeature<"ventana-veyron", "RISCVProcFamily", "VentanaVeyron",
                                         "Ventana Veyron-Series processors">;

// Assume that lock-free native-width atomics are available, even if the target
// and operating system combination would not usually provide them. The user
// is responsible for providing any necessary __sync implementations. Code
// built with this feature is not ABI-compatible with code built without this
// feature, if atomic variables are exposed across the ABI boundary.
def FeatureForcedAtomics : SubtargetFeature<
    "forced-atomics", "HasForcedAtomics", "true",
    "Assume that lock-free native-width atomics are available">;
def HasAtomicLdSt
    : Predicate<"Subtarget->hasStdExtA() || Subtarget->hasForcedAtomics()">;

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;
