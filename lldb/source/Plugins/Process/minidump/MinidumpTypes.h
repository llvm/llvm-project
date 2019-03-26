//===-- MinidumpTypes.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MinidumpTypes_h_
#define liblldb_MinidumpTypes_h_


#include "lldb/Utility/Status.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Minidump.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Endian.h"

// C includes
// C++ includes

// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms679293(v=vs.85).aspx
// https://chromium.googlesource.com/breakpad/breakpad/

namespace lldb_private {

namespace minidump {

using namespace llvm::minidump;

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

enum class CvSignature : uint32_t {
  Pdb70 = 0x53445352, // RSDS
  ElfBuildId = 0x4270454c, // BpEL (Breakpad/Crashpad minidumps)
};

// Reference:
// https://crashpad.chromium.org/doxygen/structcrashpad_1_1CodeViewRecordPDB70.html
struct CvRecordPdb70 {
  uint8_t Uuid[16];
  llvm::support::ulittle32_t Age;
  // char PDBFileName[];
};
static_assert(sizeof(CvRecordPdb70) == 20,
              "sizeof CvRecordPdb70 is not correct!");

enum class MinidumpMiscInfoFlags : uint32_t {
  ProcessID = (1 << 0),
  ProcessTimes = (1 << 1),
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ ProcessTimes)
};

template <typename T>
Status consumeObject(llvm::ArrayRef<uint8_t> &Buffer, const T *&Object) {
  Status error;
  if (Buffer.size() < sizeof(T)) {
    error.SetErrorString("Insufficient buffer!");
    return error;
  }

  Object = reinterpret_cast<const T *>(Buffer.data());
  Buffer = Buffer.drop_front(sizeof(T));
  return error;
}

// parse a MinidumpString which is with UTF-16
// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms680395(v=vs.85).aspx
llvm::Optional<std::string> parseMinidumpString(llvm::ArrayRef<uint8_t> &data);

// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms680384(v=vs.85).aspx
struct MinidumpMemoryDescriptor {
  llvm::support::ulittle64_t start_of_memory_range;
  LocationDescriptor memory;

  static llvm::ArrayRef<MinidumpMemoryDescriptor>
  ParseMemoryList(llvm::ArrayRef<uint8_t> &data);
};
static_assert(sizeof(MinidumpMemoryDescriptor) == 16,
              "sizeof MinidumpMemoryDescriptor is not correct!");

struct MinidumpMemoryDescriptor64 {
  llvm::support::ulittle64_t start_of_memory_range;
  llvm::support::ulittle64_t data_size;

  static std::pair<llvm::ArrayRef<MinidumpMemoryDescriptor64>, uint64_t>
  ParseMemory64List(llvm::ArrayRef<uint8_t> &data);
};
static_assert(sizeof(MinidumpMemoryDescriptor64) == 16,
              "sizeof MinidumpMemoryDescriptor64 is not correct!");

// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms680385(v=vs.85).aspx
struct MinidumpMemoryInfoListHeader {
  llvm::support::ulittle32_t size_of_header;
  llvm::support::ulittle32_t size_of_entry;
  llvm::support::ulittle64_t num_of_entries;
};
static_assert(sizeof(MinidumpMemoryInfoListHeader) == 16,
              "sizeof MinidumpMemoryInfoListHeader is not correct!");

enum class MinidumpMemoryInfoState : uint32_t {
  MemCommit = 0x1000,
  MemFree = 0x10000,
  MemReserve = 0x2000,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ MemFree)
};

enum class MinidumpMemoryInfoType : uint32_t {
  MemImage = 0x1000000,
  MemMapped = 0x40000,
  MemPrivate = 0x20000,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ MemImage)
};

// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/aa366786(v=vs.85).aspx
enum class MinidumpMemoryProtectionContants : uint32_t {
  PageExecute = 0x10,
  PageExecuteRead = 0x20,
  PageExecuteReadWrite = 0x40,
  PageExecuteWriteCopy = 0x80,
  PageNoAccess = 0x01,
  PageReadOnly = 0x02,
  PageReadWrite = 0x04,
  PageWriteCopy = 0x08,
  PageTargetsInvalid = 0x40000000,
  PageTargetsNoUpdate = 0x40000000,

  PageWritable = PageExecuteReadWrite | PageExecuteWriteCopy | PageReadWrite |
                 PageWriteCopy,
  PageExecutable = PageExecute | PageExecuteRead | PageExecuteReadWrite |
                   PageExecuteWriteCopy,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ PageTargetsInvalid)
};

// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms680386(v=vs.85).aspx
struct MinidumpMemoryInfo {
  llvm::support::ulittle64_t base_address;
  llvm::support::ulittle64_t allocation_base;
  llvm::support::ulittle32_t allocation_protect;
  llvm::support::ulittle32_t alignment1;
  llvm::support::ulittle64_t region_size;
  llvm::support::ulittle32_t state;
  llvm::support::ulittle32_t protect;
  llvm::support::ulittle32_t type;
  llvm::support::ulittle32_t alignment2;

  static std::vector<const MinidumpMemoryInfo *>
  ParseMemoryInfoList(llvm::ArrayRef<uint8_t> &data);

  bool isReadable() const {
    const auto mask = MinidumpMemoryProtectionContants::PageNoAccess;
    return (static_cast<uint32_t>(mask) & protect) == 0;
  }

  bool isWritable() const {
    const auto mask = MinidumpMemoryProtectionContants::PageWritable;
    return (static_cast<uint32_t>(mask) & protect) != 0;
  }

  bool isExecutable() const {
    const auto mask = MinidumpMemoryProtectionContants::PageExecutable;
    return (static_cast<uint32_t>(mask) & protect) != 0;
  }
  
  bool isMapped() const {
    return state != static_cast<uint32_t>(MinidumpMemoryInfoState::MemFree);
  }
};

static_assert(sizeof(MinidumpMemoryInfo) == 48,
              "sizeof MinidumpMemoryInfo is not correct!");

// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms680517(v=vs.85).aspx
struct MinidumpThread {
  llvm::support::ulittle32_t thread_id;
  llvm::support::ulittle32_t suspend_count;
  llvm::support::ulittle32_t priority_class;
  llvm::support::ulittle32_t priority;
  llvm::support::ulittle64_t teb;
  MinidumpMemoryDescriptor stack;
  LocationDescriptor thread_context;

  static const MinidumpThread *Parse(llvm::ArrayRef<uint8_t> &data);

  static llvm::ArrayRef<MinidumpThread>
  ParseThreadList(llvm::ArrayRef<uint8_t> &data);
};
static_assert(sizeof(MinidumpThread) == 48,
              "sizeof MinidumpThread is not correct!");

// TODO misc2, misc3 ?
// Reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms680389(v=vs.85).aspx
struct MinidumpMiscInfo {
  llvm::support::ulittle32_t size;
  // flags1 represents what info in the struct is valid
  llvm::support::ulittle32_t flags1;
  llvm::support::ulittle32_t process_id;
  llvm::support::ulittle32_t process_create_time;
  llvm::support::ulittle32_t process_user_time;
  llvm::support::ulittle32_t process_kernel_time;

  static const MinidumpMiscInfo *Parse(llvm::ArrayRef<uint8_t> &data);

  llvm::Optional<lldb::pid_t> GetPid() const;
};
static_assert(sizeof(MinidumpMiscInfo) == 24,
              "sizeof MinidumpMiscInfo is not correct!");

// The /proc/pid/status is saved as an ascii string in the file
class LinuxProcStatus {
public:
  llvm::StringRef proc_status;
  lldb::pid_t pid;

  static llvm::Optional<LinuxProcStatus> Parse(llvm::ArrayRef<uint8_t> &data);

  lldb::pid_t GetPid() const;

private:
  LinuxProcStatus() = default;
};

// MinidumpModule stuff
struct MinidumpVSFixedFileInfo {
  llvm::support::ulittle32_t signature;
  llvm::support::ulittle32_t struct_version;
  llvm::support::ulittle32_t file_version_hi;
  llvm::support::ulittle32_t file_version_lo;
  llvm::support::ulittle32_t product_version_hi;
  llvm::support::ulittle32_t product_version_lo;
  // file_flags_mask - identifies valid bits in fileFlags
  llvm::support::ulittle32_t file_flags_mask;
  llvm::support::ulittle32_t file_flags;
  llvm::support::ulittle32_t file_os;
  llvm::support::ulittle32_t file_type;
  llvm::support::ulittle32_t file_subtype;
  llvm::support::ulittle32_t file_date_hi;
  llvm::support::ulittle32_t file_date_lo;
};
static_assert(sizeof(MinidumpVSFixedFileInfo) == 52,
              "sizeof MinidumpVSFixedFileInfo is not correct!");

struct MinidumpModule {
  llvm::support::ulittle64_t base_of_image;
  llvm::support::ulittle32_t size_of_image;
  llvm::support::ulittle32_t checksum;
  llvm::support::ulittle32_t time_date_stamp;
  llvm::support::ulittle32_t module_name_rva;
  MinidumpVSFixedFileInfo version_info;
  LocationDescriptor CV_record;
  LocationDescriptor misc_record;
  llvm::support::ulittle32_t reserved0[2];
  llvm::support::ulittle32_t reserved1[2];

  static const MinidumpModule *Parse(llvm::ArrayRef<uint8_t> &data);

  static llvm::ArrayRef<MinidumpModule>
  ParseModuleList(llvm::ArrayRef<uint8_t> &data);
};
static_assert(sizeof(MinidumpModule) == 108,
              "sizeof MinidumpVSFixedFileInfo is not correct!");

// Exception stuff
struct MinidumpException {
  enum : unsigned {
    ExceptonInfoMaxParams = 15,
    DumpRequested = 0xFFFFFFFF,
  };

  llvm::support::ulittle32_t exception_code;
  llvm::support::ulittle32_t exception_flags;
  llvm::support::ulittle64_t exception_record;
  llvm::support::ulittle64_t exception_address;
  llvm::support::ulittle32_t number_parameters;
  llvm::support::ulittle32_t unused_alignment;
  llvm::support::ulittle64_t exception_information[ExceptonInfoMaxParams];
};
static_assert(sizeof(MinidumpException) == 152,
              "sizeof MinidumpException is not correct!");

struct MinidumpExceptionStream {
  llvm::support::ulittle32_t thread_id;
  llvm::support::ulittle32_t alignment;
  MinidumpException exception_record;
  LocationDescriptor thread_context;

  static const MinidumpExceptionStream *Parse(llvm::ArrayRef<uint8_t> &data);
};
static_assert(sizeof(MinidumpExceptionStream) == 168,
              "sizeof MinidumpExceptionStream is not correct!");

} // namespace minidump
} // namespace lldb_private
#endif // liblldb_MinidumpTypes_h_
