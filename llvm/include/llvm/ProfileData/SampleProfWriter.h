//===- SampleProfWriter.h - Write LLVM sample profile data ----------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions needed for writing sample profiles.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_PROFILEDATA_SAMPLEPROFWRITER_H
#define LLVM_PROFILEDATA_SAMPLEPROFWRITER_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace sampleprof {

enum SampleProfileFormat { SPF_None = 0, SPF_Text, SPF_Binary, SPF_GCC };

/// \brief Sample-based profile writer. Base class.
class SampleProfileWriter {
public:
  SampleProfileWriter(StringRef Filename, std::error_code &EC,
                      sys::fs::OpenFlags Flags)
      : OS(Filename, EC, Flags) {}
  virtual ~SampleProfileWriter() {}

  /// Write sample profiles in \p S for function \p FName.
  ///
  /// \returns status code of the file update operation.
  virtual std::error_code write(StringRef FName, const FunctionSamples &S) = 0;

  /// Write all the sample profiles in the given map of samples.
  ///
  /// \returns status code of the file update operation.
  std::error_code write(const StringMap<FunctionSamples> &ProfileMap) {
    if (std::error_code EC = writeHeader(ProfileMap))
      return EC;

    for (const auto &I : ProfileMap) {
      StringRef FName = I.first();
      const FunctionSamples &Profile = I.second;
      if (std::error_code EC = write(FName, Profile))
        return EC;
    }
    return sampleprof_error::success;
  }

  /// Profile writer factory.
  ///
  /// Create a new writer based on the value of \p Format.
  static ErrorOr<std::unique_ptr<SampleProfileWriter>>
  create(StringRef Filename, SampleProfileFormat Format);

protected:
  /// \brief Write a file header for the profile file.
  virtual std::error_code
  writeHeader(const StringMap<FunctionSamples> &ProfileMap) = 0;

  /// \brief Output stream where to emit the profile to.
  raw_fd_ostream OS;
};

/// \brief Sample-based profile writer (text format).
class SampleProfileWriterText : public SampleProfileWriter {
public:
  SampleProfileWriterText(StringRef F, std::error_code &EC)
      : SampleProfileWriter(F, EC, sys::fs::F_Text), Indent(0) {}

  std::error_code write(StringRef FName, const FunctionSamples &S) override;

protected:
  std::error_code
  writeHeader(const StringMap<FunctionSamples> &ProfileMap) override {
    return sampleprof_error::success;
  }

private:
  /// Indent level to use when writing.
  ///
  /// This is used when printing inlined callees.
  unsigned Indent;
};

/// \brief Sample-based profile writer (binary format).
class SampleProfileWriterBinary : public SampleProfileWriter {
public:
  SampleProfileWriterBinary(StringRef F, std::error_code &EC)
      : SampleProfileWriter(F, EC, sys::fs::F_None), NameTable() {}

  std::error_code write(StringRef F, const FunctionSamples &S) override;

protected:
  std::error_code
  writeHeader(const StringMap<FunctionSamples> &ProfileMap) override;
  std::error_code writeNameIdx(StringRef FName);
  std::error_code writeBody(StringRef FName, const FunctionSamples &S);

private:
  void addName(StringRef FName);
  void addNames(const FunctionSamples &S);

  MapVector<StringRef, uint32_t> NameTable;
};

} // End namespace sampleprof

} // End namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROFWRITER_H
