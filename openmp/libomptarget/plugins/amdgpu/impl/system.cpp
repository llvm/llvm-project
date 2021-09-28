//===--- amdgpu/impl/system.cpp ----------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <libelf.h>

#include <cassert>
#include <sstream>
#include <string>

#include "internal.h"
#include "rt.h"

#include "msgpack.h"

namespace hsa {
// Wrap HSA iterate API in a shim that allows passing general callables
template <typename C>
hsa_status_t executable_iterate_symbols(hsa_executable_t executable, C cb) {
  auto L = [](hsa_executable_t executable, hsa_executable_symbol_t symbol,
              void *data) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(data);
    return (*unwrapped)(executable, symbol);
  };
  return hsa_executable_iterate_symbols(executable, L,
                                        static_cast<void *>(&cb));
}
} // namespace hsa

typedef unsigned char *address;
/*
 * Note descriptors.
 */
// FreeBSD already declares Elf_Note (indirectly via <libelf.h>)
#if !defined(__FreeBSD__)
typedef struct {
  uint32_t n_namesz; /* Length of note's name. */
  uint32_t n_descsz; /* Length of note's value. */
  uint32_t n_type;   /* Type of note. */
  // then name
  // then padding, optional
  // then desc, at 4 byte alignment (not 8, despite being elf64)
} Elf_Note;
#endif

// The following include file and following structs/enums
// have been replicated on a per-use basis below. For example,
// llvm::AMDGPU::HSAMD::Kernel::Metadata has several fields,
// but we may care only about kernargSegmentSize_ for now, so
// we just include that field in our KernelMD implementation. We
// chose this approach to replicate in order to avoid forcing
// a dependency on LLVM_INCLUDE_DIR just to compile the runtime.
// #include "llvm/Support/AMDGPUMetadata.h"
// typedef llvm::AMDGPU::HSAMD::Metadata CodeObjectMD;
// typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
// typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;
// using llvm::AMDGPU::HSAMD::AccessQualifier;
// using llvm::AMDGPU::HSAMD::AddressSpaceQualifier;
// using llvm::AMDGPU::HSAMD::ValueKind;
// using llvm::AMDGPU::HSAMD::ValueType;

class KernelArgMD {
public:
  enum class ValueKind {
    HiddenGlobalOffsetX,
    HiddenGlobalOffsetY,
    HiddenGlobalOffsetZ,
    HiddenNone,
    HiddenPrintfBuffer,
    HiddenDefaultQueue,
    HiddenCompletionAction,
    HiddenMultiGridSyncArg,
    HiddenHostcallBuffer,
    Unknown
  };

  KernelArgMD()
      : name_(std::string()), typeName_(std::string()), size_(0), offset_(0),
        align_(0), valueKind_(ValueKind::Unknown) {}

  // fields
  std::string name_;
  std::string typeName_;
  uint32_t size_;
  uint32_t offset_;
  uint32_t align_;
  ValueKind valueKind_;
};

class KernelMD {
public:
  KernelMD() : kernargSegmentSize_(0ull) {}

  // fields
  uint64_t kernargSegmentSize_;
};

static const std::map<std::string, KernelArgMD::ValueKind> ArgValueKind = {
    //    Including only those fields that are relevant to the runtime.
    //    {"ByValue", KernelArgMD::ValueKind::ByValue},
    //    {"GlobalBuffer", KernelArgMD::ValueKind::GlobalBuffer},
    //    {"DynamicSharedPointer",
    //    KernelArgMD::ValueKind::DynamicSharedPointer},
    //    {"Sampler", KernelArgMD::ValueKind::Sampler},
    //    {"Image", KernelArgMD::ValueKind::Image},
    //    {"Pipe", KernelArgMD::ValueKind::Pipe},
    //    {"Queue", KernelArgMD::ValueKind::Queue},
    {"HiddenGlobalOffsetX", KernelArgMD::ValueKind::HiddenGlobalOffsetX},
    {"HiddenGlobalOffsetY", KernelArgMD::ValueKind::HiddenGlobalOffsetY},
    {"HiddenGlobalOffsetZ", KernelArgMD::ValueKind::HiddenGlobalOffsetZ},
    {"HiddenNone", KernelArgMD::ValueKind::HiddenNone},
    {"HiddenPrintfBuffer", KernelArgMD::ValueKind::HiddenPrintfBuffer},
    {"HiddenDefaultQueue", KernelArgMD::ValueKind::HiddenDefaultQueue},
    {"HiddenCompletionAction", KernelArgMD::ValueKind::HiddenCompletionAction},
    {"HiddenMultiGridSyncArg", KernelArgMD::ValueKind::HiddenMultiGridSyncArg},
    {"HiddenHostcallBuffer", KernelArgMD::ValueKind::HiddenHostcallBuffer},
    // v3
    //    {"by_value", KernelArgMD::ValueKind::ByValue},
    //    {"global_buffer", KernelArgMD::ValueKind::GlobalBuffer},
    //    {"dynamic_shared_pointer",
    //    KernelArgMD::ValueKind::DynamicSharedPointer},
    //    {"sampler", KernelArgMD::ValueKind::Sampler},
    //    {"image", KernelArgMD::ValueKind::Image},
    //    {"pipe", KernelArgMD::ValueKind::Pipe},
    //    {"queue", KernelArgMD::ValueKind::Queue},
    {"hidden_global_offset_x", KernelArgMD::ValueKind::HiddenGlobalOffsetX},
    {"hidden_global_offset_y", KernelArgMD::ValueKind::HiddenGlobalOffsetY},
    {"hidden_global_offset_z", KernelArgMD::ValueKind::HiddenGlobalOffsetZ},
    {"hidden_none", KernelArgMD::ValueKind::HiddenNone},
    {"hidden_printf_buffer", KernelArgMD::ValueKind::HiddenPrintfBuffer},
    {"hidden_default_queue", KernelArgMD::ValueKind::HiddenDefaultQueue},
    {"hidden_completion_action",
     KernelArgMD::ValueKind::HiddenCompletionAction},
    {"hidden_multigrid_sync_arg",
     KernelArgMD::ValueKind::HiddenMultiGridSyncArg},
    {"hidden_hostcall_buffer", KernelArgMD::ValueKind::HiddenHostcallBuffer},
};

namespace core {

hsa_status_t callbackEvent(const hsa_amd_event_t *event, void *data) {
  if (event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT) {
    hsa_amd_gpu_memory_fault_info_t memory_fault = event->memory_fault;
    // memory_fault.agent
    // memory_fault.virtual_address
    // memory_fault.fault_reason_mask
    // fprintf("[GPU Error at %p: Reason is ", memory_fault.virtual_address);
    std::stringstream stream;
    stream << std::hex << (uintptr_t)memory_fault.virtual_address;
    std::string addr("0x" + stream.str());

    std::string err_string = "[GPU Memory Error] Addr: " + addr;
    err_string += " Reason: ";
    if (!(memory_fault.fault_reason_mask & 0x00111111)) {
      err_string += "No Idea! ";
    } else {
      if (memory_fault.fault_reason_mask & 0x00000001)
        err_string += "Page not present or supervisor privilege. ";
      if (memory_fault.fault_reason_mask & 0x00000010)
        err_string += "Write access to a read-only page. ";
      if (memory_fault.fault_reason_mask & 0x00000100)
        err_string += "Execute access to a page marked NX. ";
      if (memory_fault.fault_reason_mask & 0x00001000)
        err_string += "Host access only. ";
      if (memory_fault.fault_reason_mask & 0x00010000)
        err_string += "ECC failure (if supported by HW). ";
      if (memory_fault.fault_reason_mask & 0x00100000)
        err_string += "Can't determine the exact fault address. ";
    }
    fprintf(stderr, "%s\n", err_string.c_str());
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t atl_init_gpu_context() {
  hsa_status_t err = hsa_amd_register_system_event_handler(callbackEvent, NULL);
  if (err != HSA_STATUS_SUCCESS) {
    printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Registering the system for memory faults", get_error_string(err));
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

static bool isImplicit(KernelArgMD::ValueKind value_kind) {
  switch (value_kind) {
  case KernelArgMD::ValueKind::HiddenGlobalOffsetX:
  case KernelArgMD::ValueKind::HiddenGlobalOffsetY:
  case KernelArgMD::ValueKind::HiddenGlobalOffsetZ:
  case KernelArgMD::ValueKind::HiddenNone:
  case KernelArgMD::ValueKind::HiddenPrintfBuffer:
  case KernelArgMD::ValueKind::HiddenDefaultQueue:
  case KernelArgMD::ValueKind::HiddenCompletionAction:
  case KernelArgMD::ValueKind::HiddenMultiGridSyncArg:
  case KernelArgMD::ValueKind::HiddenHostcallBuffer:
    return true;
  default:
    return false;
  }
}

static std::pair<unsigned char *, unsigned char *>
find_metadata(void *binary, size_t binSize) {
  std::pair<unsigned char *, unsigned char *> failure = {nullptr, nullptr};

  Elf *e = elf_memory(static_cast<char *>(binary), binSize);
  if (elf_kind(e) != ELF_K_ELF) {
    return failure;
  }

  size_t numpHdrs;
  if (elf_getphdrnum(e, &numpHdrs) != 0) {
    return failure;
  }

  Elf64_Phdr *pHdrs = elf64_getphdr(e);
  for (size_t i = 0; i < numpHdrs; ++i) {
    Elf64_Phdr pHdr = pHdrs[i];

    // Look for the runtime metadata note
    if (pHdr.p_type == PT_NOTE && pHdr.p_align >= sizeof(int)) {
      // Iterate over the notes in this segment
      address ptr = (address)binary + pHdr.p_offset;
      address segmentEnd = ptr + pHdr.p_filesz;

      while (ptr < segmentEnd) {
        Elf_Note *note = reinterpret_cast<Elf_Note *>(ptr);
        address name = (address)&note[1];

        if (note->n_type == 7 || note->n_type == 8) {
          return failure;
        } else if (note->n_type == 10 /* NT_AMD_AMDGPU_HSA_METADATA */ &&
                   note->n_namesz == sizeof "AMD" &&
                   !memcmp(name, "AMD", note->n_namesz)) {
          // code object v2 uses yaml metadata, no longer supported
          return failure;
        } else if (note->n_type == 32 /* NT_AMDGPU_METADATA */ &&
                   note->n_namesz == sizeof "AMDGPU" &&
                   !memcmp(name, "AMDGPU", note->n_namesz)) {

          // n_descsz = 485
          // value is padded to 4 byte alignment, may want to move end up to
          // match
          size_t offset = sizeof(uint32_t) * 3 /* fields */
                          + sizeof("AMDGPU")   /* name */
                          + 1 /* padding to 4 byte alignment */;

          // Including the trailing padding means both pointers are 4 bytes
          // aligned, which may be useful later.
          unsigned char *metadata_start = (unsigned char *)ptr + offset;
          unsigned char *metadata_end =
              metadata_start + core::alignUp(note->n_descsz, 4);
          return {metadata_start, metadata_end};
        }
        ptr += sizeof(*note) + core::alignUp(note->n_namesz, sizeof(int)) +
               core::alignUp(note->n_descsz, sizeof(int));
      }
    }
  }

  return failure;
}

namespace {
int map_lookup_array(msgpack::byte_range message, const char *needle,
                     msgpack::byte_range *res, uint64_t *size) {
  unsigned count = 0;
  struct s : msgpack::functors_defaults<s> {
    s(unsigned &count, uint64_t *size) : count(count), size(size) {}
    unsigned &count;
    uint64_t *size;
    const unsigned char *handle_array(uint64_t N, msgpack::byte_range bytes) {
      count++;
      *size = N;
      return bytes.end;
    }
  };

  msgpack::foreach_map(message,
                       [&](msgpack::byte_range key, msgpack::byte_range value) {
                         if (msgpack::message_is_string(key, needle)) {
                           // If the message is an array, record number of
                           // elements in *size
                           msgpack::handle_msgpack<s>(value, {count, size});
                           // return the whole array
                           *res = value;
                         }
                       });
  // Only claim success if exactly one key/array pair matched
  return count != 1;
}

int map_lookup_string(msgpack::byte_range message, const char *needle,
                      std::string *res) {
  unsigned count = 0;
  struct s : public msgpack::functors_defaults<s> {
    s(unsigned &count, std::string *res) : count(count), res(res) {}
    unsigned &count;
    std::string *res;
    void handle_string(size_t N, const unsigned char *str) {
      count++;
      *res = std::string(str, str + N);
    }
  };
  msgpack::foreach_map(message,
                       [&](msgpack::byte_range key, msgpack::byte_range value) {
                         if (msgpack::message_is_string(key, needle)) {
                           msgpack::handle_msgpack<s>(value, {count, res});
                         }
                       });
  return count != 1;
}

int map_lookup_uint64_t(msgpack::byte_range message, const char *needle,
                        uint64_t *res) {
  unsigned count = 0;
  msgpack::foreach_map(message,
                       [&](msgpack::byte_range key, msgpack::byte_range value) {
                         if (msgpack::message_is_string(key, needle)) {
                           msgpack::foronly_unsigned(value, [&](uint64_t x) {
                             count++;
                             *res = x;
                           });
                         }
                       });
  return count != 1;
}

int array_lookup_element(msgpack::byte_range message, uint64_t elt,
                         msgpack::byte_range *res) {
  int rc = 1;
  uint64_t i = 0;
  msgpack::foreach_array(message, [&](msgpack::byte_range value) {
    if (i == elt) {
      *res = value;
      rc = 0;
    }
    i++;
  });
  return rc;
}

int populate_kernelArgMD(msgpack::byte_range args_element,
                         KernelArgMD *kernelarg) {
  using namespace msgpack;
  int error = 0;
  foreach_map(args_element, [&](byte_range key, byte_range value) -> void {
    if (message_is_string(key, ".name")) {
      foronly_string(value, [&](size_t N, const unsigned char *str) {
        kernelarg->name_ = std::string(str, str + N);
      });
    } else if (message_is_string(key, ".type_name")) {
      foronly_string(value, [&](size_t N, const unsigned char *str) {
        kernelarg->typeName_ = std::string(str, str + N);
      });
    } else if (message_is_string(key, ".size")) {
      foronly_unsigned(value, [&](uint64_t x) { kernelarg->size_ = x; });
    } else if (message_is_string(key, ".offset")) {
      foronly_unsigned(value, [&](uint64_t x) { kernelarg->offset_ = x; });
    } else if (message_is_string(key, ".value_kind")) {
      foronly_string(value, [&](size_t N, const unsigned char *str) {
        std::string s = std::string(str, str + N);
        auto itValueKind = ArgValueKind.find(s);
        if (itValueKind != ArgValueKind.end()) {
          kernelarg->valueKind_ = itValueKind->second;
        }
      });
    }
  });
  return error;
}
} // namespace

static hsa_status_t get_code_object_custom_metadata(
    void *binary, size_t binSize,
    std::map<std::string, atl_kernel_info_t> &KernelInfoTable) {
  // parse code object with different keys from v2
  // also, the kernel name is not the same as the symbol name -- so a
  // symbol->name map is needed

  std::pair<unsigned char *, unsigned char *> metadata =
      find_metadata(binary, binSize);
  if (!metadata.first) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  uint64_t kernelsSize = 0;
  int msgpack_errors = 0;
  msgpack::byte_range kernel_array;
  msgpack_errors =
      map_lookup_array({metadata.first, metadata.second}, "amdhsa.kernels",
                       &kernel_array, &kernelsSize);
  if (msgpack_errors != 0) {
    printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
           "kernels lookup in program metadata");
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  for (size_t i = 0; i < kernelsSize; i++) {
    assert(msgpack_errors == 0);
    std::string kernelName;
    std::string symbolName;

    msgpack::byte_range element;
    msgpack_errors += array_lookup_element(kernel_array, i, &element);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "element lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    msgpack_errors += map_lookup_string(element, ".name", &kernelName);
    msgpack_errors += map_lookup_string(element, ".symbol", &symbolName);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "strings lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    // Make sure that kernelName + ".kd" == symbolName
    if ((kernelName + ".kd") != symbolName) {
      printf("[%s:%d] Kernel name mismatching symbol: %s != %s + .kd\n",
             __FILE__, __LINE__, symbolName.c_str(), kernelName.c_str());
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    atl_kernel_info_t info = {0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}};

    uint64_t sgpr_count, vgpr_count, sgpr_spill_count, vgpr_spill_count;
    msgpack_errors += map_lookup_uint64_t(element, ".sgpr_count", &sgpr_count);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "sgpr count metadata lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    info.sgpr_count = sgpr_count;

    msgpack_errors += map_lookup_uint64_t(element, ".vgpr_count", &vgpr_count);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "vgpr count metadata lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    info.vgpr_count = vgpr_count;

    msgpack_errors +=
        map_lookup_uint64_t(element, ".sgpr_spill_count", &sgpr_spill_count);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "sgpr spill count metadata lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    info.sgpr_spill_count = sgpr_spill_count;

    msgpack_errors +=
        map_lookup_uint64_t(element, ".vgpr_spill_count", &vgpr_spill_count);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "vgpr spill count metadata lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    info.vgpr_spill_count = vgpr_spill_count;

    size_t kernel_explicit_args_size = 0;
    uint64_t kernel_segment_size;
    msgpack_errors += map_lookup_uint64_t(element, ".kernarg_segment_size",
                                          &kernel_segment_size);
    if (msgpack_errors != 0) {
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
             "kernarg segment size metadata lookup in kernel metadata");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    bool hasHiddenArgs = false;
    if (kernel_segment_size > 0) {
      uint64_t argsSize;
      size_t offset = 0;

      msgpack::byte_range args_array;
      msgpack_errors +=
          map_lookup_array(element, ".args", &args_array, &argsSize);
      if (msgpack_errors != 0) {
        printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
               "kernel args metadata lookup in kernel metadata");
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }

      info.num_args = argsSize;

      for (size_t i = 0; i < argsSize; ++i) {
        KernelArgMD lcArg;

        msgpack::byte_range args_element;
        msgpack_errors += array_lookup_element(args_array, i, &args_element);
        if (msgpack_errors != 0) {
          printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
                 "iterate args map in kernel args metadata");
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
        }

        msgpack_errors += populate_kernelArgMD(args_element, &lcArg);
        if (msgpack_errors != 0) {
          printf("[%s:%d] %s failed\n", __FILE__, __LINE__,
                 "iterate args map in kernel args metadata");
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
        }
        // populate info with sizes and offsets
        info.arg_sizes.push_back(lcArg.size_);
        // v3 has offset field and not align field
        size_t new_offset = lcArg.offset_;
        size_t padding = new_offset - offset;
        offset = new_offset;
        info.arg_offsets.push_back(lcArg.offset_);
        DP("Arg[%lu] \"%s\" (%u, %u)\n", i, lcArg.name_.c_str(), lcArg.size_,
           lcArg.offset_);
        offset += lcArg.size_;

        // check if the arg is a hidden/implicit arg
        // this logic assumes that all hidden args are 8-byte aligned
        if (!isImplicit(lcArg.valueKind_)) {
          kernel_explicit_args_size += lcArg.size_;
        } else {
          hasHiddenArgs = true;
        }
        kernel_explicit_args_size += padding;
      }
    }

    // add size of implicit args, e.g.: offset x, y and z and pipe pointer, but
    // do not count the compiler set implicit args, but set your own implicit
    // args by discounting the compiler set implicit args
    info.kernel_segment_size =
        (hasHiddenArgs ? kernel_explicit_args_size : kernel_segment_size) +
        sizeof(impl_implicit_args_t);
    DP("[%s: kernarg seg size] (%lu --> %u)\n", kernelName.c_str(),
       kernel_segment_size, info.kernel_segment_size);

    // kernel received, now add it to the kernel info table
    KernelInfoTable[kernelName] = info;
  }

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t
populate_InfoTables(hsa_executable_symbol_t symbol,
                    std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
                    std::map<std::string, atl_symbol_info_t> &SymbolInfoTable) {
  hsa_symbol_kind_t type;

  uint32_t name_length;
  hsa_status_t err;
  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                                       &type);
  if (err != HSA_STATUS_SUCCESS) {
    printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Symbol info extraction", get_error_string(err));
    return err;
  }
  DP("Exec Symbol type: %d\n", type);
  if (type == HSA_SYMBOL_KIND_KERNEL) {
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Symbol info extraction", get_error_string(err));
      return err;
    }
    char *name = reinterpret_cast<char *>(malloc(name_length + 1));
    err = hsa_executable_symbol_get_info(symbol,
                                         HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Symbol info extraction", get_error_string(err));
      return err;
    }
    // remove the suffix .kd from symbol name.
    name[name_length - 3] = 0;

    atl_kernel_info_t info;
    std::string kernelName(name);
    // by now, the kernel info table should already have an entry
    // because the non-ROCr custom code object parsing is called before
    // iterating over the code object symbols using ROCr
    if (KernelInfoTable.find(kernelName) == KernelInfoTable.end()) {
      DP("amdgpu internal consistency error\n");
      return HSA_STATUS_ERROR;
    }
    // found, so assign and update
    info = KernelInfoTable[kernelName];

    /* Extract dispatch information from the symbol */
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &(info.kernel_object));
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Extracting the symbol from the executable",
             get_error_string(err));
      return err;
    }
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &(info.group_segment_size));
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Extracting the group segment size from the executable",
             get_error_string(err));
      return err;
    }
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &(info.private_segment_size));
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Extracting the private segment from the executable",
             get_error_string(err));
      return err;
    }

    DP("Kernel %s --> %lx symbol %u group segsize %u pvt segsize %u bytes "
       "kernarg\n",
       kernelName.c_str(), info.kernel_object, info.group_segment_size,
       info.private_segment_size, info.kernel_segment_size);

    // assign it back to the kernel info table
    KernelInfoTable[kernelName] = info;
    free(name);
  } else if (type == HSA_SYMBOL_KIND_VARIABLE) {
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Symbol info extraction", get_error_string(err));
      return err;
    }
    char *name = reinterpret_cast<char *>(malloc(name_length + 1));
    err = hsa_executable_symbol_get_info(symbol,
                                         HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Symbol info extraction", get_error_string(err));
      return err;
    }
    name[name_length] = 0;

    atl_symbol_info_t info;

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &(info.addr));
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Symbol info address extraction", get_error_string(err));
      return err;
    }

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &(info.size));
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Symbol info size extraction", get_error_string(err));
      return err;
    }

    DP("Symbol %s = %p (%u bytes)\n", name, (void *)info.addr, info.size);
    SymbolInfoTable[std::string(name)] = info;
    free(name);
  } else {
    DP("Symbol is an indirect function\n");
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t RegisterModuleFromMemory(
    std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
    std::map<std::string, atl_symbol_info_t> &SymbolInfoTable,
    void *module_bytes, size_t module_size, hsa_agent_t agent,
    hsa_status_t (*on_deserialized_data)(void *data, size_t size,
                                         void *cb_state),
    void *cb_state, std::vector<hsa_executable_t> &HSAExecutables) {
  hsa_status_t err;
  hsa_executable_t executable = {0};
  hsa_profile_t agent_profile;

  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_profile);
  if (err != HSA_STATUS_SUCCESS) {
    printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Query the agent profile", get_error_string(err));
    return HSA_STATUS_ERROR;
  }
  // FIXME: Assume that every profile is FULL until we understand how to build
  // GCN with base profile
  agent_profile = HSA_PROFILE_FULL;
  /* Create the empty executable.  */
  err = hsa_executable_create(agent_profile, HSA_EXECUTABLE_STATE_UNFROZEN, "",
                              &executable);
  if (err != HSA_STATUS_SUCCESS) {
    printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Create the executable", get_error_string(err));
    return HSA_STATUS_ERROR;
  }

  bool module_load_success = false;
  do // Existing control flow used continue, preserve that for this patch
  {
    {
      // Some metadata info is not available through ROCr API, so use custom
      // code object metadata parsing to collect such metadata info

      err = get_code_object_custom_metadata(module_bytes, module_size,
                                            KernelInfoTable);
      if (err != HSA_STATUS_SUCCESS) {
        DP("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Getting custom code object metadata", get_error_string(err));
        continue;
      }

      // Deserialize code object.
      hsa_code_object_t code_object = {0};
      err = hsa_code_object_deserialize(module_bytes, module_size, NULL,
                                        &code_object);
      if (err != HSA_STATUS_SUCCESS) {
        DP("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Code Object Deserialization", get_error_string(err));
        continue;
      }
      assert(0 != code_object.handle);

      // Mutating the device image here avoids another allocation & memcpy
      void *code_object_alloc_data =
          reinterpret_cast<void *>(code_object.handle);
      hsa_status_t impl_err =
          on_deserialized_data(code_object_alloc_data, module_size, cb_state);
      if (impl_err != HSA_STATUS_SUCCESS) {
        printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
               "Error in deserialized_data callback",
               get_error_string(impl_err));
        return impl_err;
      }

      /* Load the code object.  */
      err =
          hsa_executable_load_code_object(executable, agent, code_object, NULL);
      if (err != HSA_STATUS_SUCCESS) {
        DP("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Loading the code object", get_error_string(err));
        continue;
      }

      // cannot iterate over symbols until executable is frozen
    }
    module_load_success = true;
  } while (0);
  DP("Modules loaded successful? %d\n", module_load_success);
  if (module_load_success) {
    /* Freeze the executable; it can now be queried for symbols.  */
    err = hsa_executable_freeze(executable, "");
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Freeze the executable", get_error_string(err));
      return HSA_STATUS_ERROR;
    }

    err = hsa::executable_iterate_symbols(
        executable,
        [&](hsa_executable_t, hsa_executable_symbol_t symbol) -> hsa_status_t {
          return populate_InfoTables(symbol, KernelInfoTable, SymbolInfoTable);
        });
    if (err != HSA_STATUS_SUCCESS) {
      printf("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
             "Iterating over symbols for execuatable", get_error_string(err));
      return HSA_STATUS_ERROR;
    }

    // save the executable and destroy during finalize
    HSAExecutables.push_back(executable);
    return HSA_STATUS_SUCCESS;
  } else {
    return HSA_STATUS_ERROR;
  }
}

} // namespace core
