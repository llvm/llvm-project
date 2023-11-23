//===- llvm-elf2bin.h - Header file for llvm-elf2bin ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/Twine.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"

/*
 * A structure that describes a single loadable segment in an ELF
 * image. 'fileoffset' and 'filesize' designate a region of bytes in
 * the file; 'baseaddr' specifies the memory address that region is
 * expected to be loaded at.
 *
 * 'memsize' indicates how many bytes the region will occupy once it's
 * loaded. This should never be less than 'filesize'. If it's more
 * than 'filesize', it indicates that zero padding should be appended
 * by the loader to pad to the full size.
 */
struct Segment {
  uint64_t baseaddr, fileoffset, filesize, memsize;
};

/*
 * A structure containing the name of an input file to the program,
 * and the result of loading it into an ELFObjectFile.
 */
struct InputObject {
  std::string filename;
  std::unique_ptr<llvm::MemoryBuffer> membuf;
  std::unique_ptr<llvm::object::ELFObjectFileBase> elf;

  /*
   * List the loadable segments in the file.
   *
   * The flag 'physical' indicates that the 'baseaddr' value for each
   * returned segment should be the physical address of that segment,
   * i.e. the p_paddr field in its program header table entry. If it's
   * false, the virtual address will be used instead, i.e. the p_vaddr
   * field.
   */
  std::vector<Segment> segments(bool physical);

  uint64_t entry_point();
};

/*
 * Write a single binary or Verilog hex file. (A Verilog hex file is
 * just the plain binary data, represented as a sequence of text lines
 * each containing a hex-encoded byte).
 *
 * Input data comes from the file 'infile', at offset 'fileoffset',
 * 'size' bytes long. 'zi_size' zero bytes are appended after that.
 *
 * Bank switching is supported by the bank_* parameters, which select
 * a subset of the input bytes to be written to the output.
 * Specifically, each input byte is included or excluded depending on
 * the residue of its position in the input, mod 'bank_modulus'.
 * 'bank_nres' consecutive residues are kept, starting at
 * 'bank_firstres'. For example:
 *
 * modulus=8, firstres=0, nres=1: keep only the bytes whose positions
 * are congruent to 0 mod 8. (Just one residue, namely 0.) That is,
 * divide the input into 8-byte blocks and only write the initial byte
 * of each block.
 *
 * modulus=8, firstres=0, nres=2: keep the bytes whose positions are
 * congruent to {0,1} mod 8. (Two consecutive residues, starting at 0.)
 *
 * modulus=8, firstres=4, nres=2: keep the bytes whose positions are
 * congruent to {4,5} mod 8. (Still two residues, but now starting at 4.)
 *
 * If no bank switching is needed, then modulus=1, firstres=0, nres=1
 * is a combination that indicates 'write all input bytes to the output'.
 *
 * The output is written to the file 'outfile'.
 *
 * These functions will exit the entire program with an error message if
 * anything goes wrong. So callers need not handle the failure case.
 */
void bin_write(InputObject &inobj, const std::string &outfile,
               uint64_t fileoffset, uint64_t size, uint64_t zi_size,
               uint64_t bank_modulus, uint64_t bank_firstres,
               uint64_t bank_nres);
void vhx_write(InputObject &inobj, const std::string &outfile,
               uint64_t fileoffset, uint64_t size, uint64_t zi_size,
               uint64_t bank_modulus, uint64_t bank_firstres,
               uint64_t bank_nres);

/*
 * Write a combined binary or Verilog hex file, including multiple
 * segments from an ELF file, at their correct relative offsets.
 *
 * 'segments' gives the list of segments from the ELF file to include.
 * Each Segment structure includes the file offset, base address and
 * size, so these functions can work out the padding required in
 * between.
 *
 * 'baseaddr' gives the address corresponding to the start of the
 * file. (So if this is lower than the base address of the first
 * segment, then padding must be inserted at the very start of the
 * file.)
 *
 * If 'include_zi' is set, then the ZI padding specified in the ELF
 * file after each segment will be reliably included in the output
 * file. (This is likely only relevant to the final segment, because
 * if there are two segments with space between them, then the ZI
 * padding for the first segment will occupy some of that space, and
 * will be included in the file anyway.) If 'include_zi' is false, the
 * output file will end as soon as the last byte of actual file data
 * has been written.
 *
 * The bank_* parameters are interpreted identically to the previous
 * pair of functions. So is 'outfile'.
 *
 * These functions too will exit with an error in case of failure.
 */
void bincombined_write(InputObject &inobj, const std::string &outfile,
                       const std::vector<Segment> &segments, bool include_zi,
                       std::optional<uint64_t> baseaddr, uint64_t bank_modulus,
                       uint64_t bank_firstres, uint64_t bank_nres);
void vhxcombined_write(InputObject &inobj, const std::string &outfile,
                       const std::vector<Segment> &segments, bool include_zi,
                       std::optional<uint64_t> baseaddr, uint64_t bank_modulus,
                       uint64_t bank_firstres, uint64_t bank_nres);

/*
 * Write a structured hex file (Intel Hex or Motorola S-records)
 * describing an ELF image.
 *
 * 'segments' lists the loadable segments that should be included
 * (which may not be all the segments in the file, if the command line
 * specified only a subset of them).
 *
 * 'include_zi' indicates whether the ZI padding at the end of each
 * segment should be explicitly represented in the hex file.
 *
 * 'datareclen' indicates how many bytes of data should appear in each
 * hex record. (Fewer bytes per record mean more file size overhead,
 * but also less likelihood of overflowing a reader's size limit.
 *
 * These functions will exit the entire program with an error message if
 * anything goes wrong. So callers need not handle the failure case.
 */
void ihex_write(InputObject &inobj, const std::string &outfile,
                const std::vector<Segment> &segments, bool include_zi,
                uint64_t datareclen);
void srec_write(InputObject &inobj, const std::string &outfile,
                const std::vector<Segment> &segments, bool include_zi,
                uint64_t datareclen);

/*
 * Error-reporting functions. These are all fatal.
 */
[[noreturn]] void fatal(llvm::StringRef filename, llvm::Twine message, llvm::Error err);
[[noreturn]] void fatal(llvm::StringRef filename, llvm::Twine message);
[[noreturn]] void fatal(InputObject &inobj, llvm::Twine message,
                        llvm::Error err);
[[noreturn]] void fatal(InputObject &inobj, llvm::Twine message);
[[noreturn]] void fatal(llvm::Twine message);
