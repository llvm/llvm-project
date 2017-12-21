//===--- DataBufferLLVM.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/DataBufferLLVM.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <assert.h>    // for assert
#include <type_traits> // for move

using namespace lldb_private;

DataBufferLLVM::DataBufferLLVM(
    std::unique_ptr<llvm::WritableMemoryBuffer> MemBuffer)
    : Buffer(std::move(MemBuffer)) {
  assert(Buffer != nullptr &&
         "Cannot construct a DataBufferLLVM with a null buffer");
}

DataBufferLLVM::~DataBufferLLVM() {}

std::shared_ptr<DataBufferLLVM>
DataBufferLLVM::CreateSliceFromPath(const llvm::Twine &Path, uint64_t Size,
                                    uint64_t Offset) {
  // If the file resides non-locally, pass the volatile flag so that we don't
  // mmap it.
  bool IsVolatile = !llvm::sys::fs::is_local(Path);

  auto Buffer =
      llvm::WritableMemoryBuffer::getFileSlice(Path, Size, Offset, IsVolatile);
  if (!Buffer)
    return nullptr;
  return std::shared_ptr<DataBufferLLVM>(
      new DataBufferLLVM(std::move(*Buffer)));
}

std::shared_ptr<DataBufferLLVM>
DataBufferLLVM::CreateFromPath(const llvm::Twine &Path) {
  // If the file resides non-locally, pass the volatile flag so that we don't
  // mmap it.
  bool IsVolatile = !llvm::sys::fs::is_local(Path);

  auto Buffer = llvm::WritableMemoryBuffer::getFile(Path, -1, IsVolatile);
  if (!Buffer)
    return nullptr;
  return std::shared_ptr<DataBufferLLVM>(
      new DataBufferLLVM(std::move(*Buffer)));
}

uint8_t *DataBufferLLVM::GetBytes() {
  return reinterpret_cast<uint8_t *>(Buffer->getBufferStart());
}

const uint8_t *DataBufferLLVM::GetBytes() const {
  return reinterpret_cast<const uint8_t *>(Buffer->getBufferStart());
}

lldb::offset_t DataBufferLLVM::GetByteSize() const {
  return Buffer->getBufferSize();
}
